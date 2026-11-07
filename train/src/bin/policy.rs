#[path = "../inputs.rs"]
mod inputs;
#[path = "../loader.rs"]
mod loader;

use bullet_compiler::{
    ir::NodeId,
    model::{Layout, MType, ModelBuilder, ModelOperation},
    tensor::{
        DType, IRTrace, TNode, TType, TValue, TensorIR,
        operation::autograd::{CustomAutograd, CustomAutogradOp},
    },
};
use bullet_gpu::{
    kernel::KernelSrc,
    runtime::{Dim3, Gpu},
};
use bullet_lib::{
    game::inputs::SparseInputType,
    nn::MatmulPrecision,
    trainer::schedule::lr::{self, LrScheduler},
    value::save::save_to_checkpoint,
};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelWeights, SavedFormat},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::DataReader,
    run::{
        DataLoader, DataLoadingError, DefaultDevice, HostPool, PreparedBatchHost, Step, TrainingSchedule,
        TrainingSteps, train,
    },
};
use inputs::*;
use loader::TakReader;
use rand::{Rng, rng};
use std::sync::Arc;
use syntaks::core::{Score, is_decisive, normalizer};
use syntaks::eval::{
    nnue::{L1_SIZE as L1, Q0, packus_position},
    policy::{DIVISOR as POLICY_DIVISOR, MOVES, move_index},
};
use syntaks::format::GameResult;
use syntaks::movegen::generate_moves;
use syntaks::position::Position;
use syntaks::takmove::Move;

const LEGAL_SLOTS: usize = 128;
const POLICY_WEIGHT_SCALE: f32 = POLICY_DIVISOR / 127.0;

const MAP_THREADS: usize = 16;
const LOAD_THREADS: usize = 16;
const SHUFFLE_BUFFER: usize = 4096;
const THREADS: usize = 256;

const LOSS_KERNEL: &str = include_str!("../policy_loss.cuh");

#[derive(Debug, PartialEq, Eq)]
struct LegalLossAutograd {
    batch: usize,
}

impl LegalLossAutograd {
    fn run_kernel<'a>(
        &self,
        name: &str,
        inputs: &[TNode<'a>],
        output: TType,
        zeroed: bool,
    ) -> Result<TNode<'a>, IRTrace> {
        let batch = self.batch;
        let source = format!("#define THREADS {THREADS}\n#define BATCH {batch}\n#define MOVES {MOVES}\n{LOSS_KERNEL}");
        let args = (0..inputs.len()).map(|i| (i, true)).chain([(0, false)]).collect();
        let gdim = Dim3 {
            x: batch.div_ceil(THREADS).try_into().unwrap(),
            y: 1,
            z: 1,
        };
        let kernel = unsafe {
            KernelSrc::new(
                inputs.iter().map(TNode::ty).collect(),
                vec![output],
                name.to_string(),
                source,
                args,
                if zeroed { [0].into() } else { Default::default() },
                gdim,
                THREADS as u32,
                0,
            )
        };
        inputs[0].builder().add_op(inputs, kernel).map(|nodes| nodes[0])
    }
}

impl CustomAutograd for LegalLossAutograd {
    fn opname(&self) -> String {
        format!("legal_loss<{}>", self.batch)
    }

    fn inputs(&self) -> Vec<TType> {
        vec![
            TType::new(self.batch * MOVES, DType::F32),
            TType::new(self.batch * LEGAL_SLOTS, DType::F32),
            TType::new(self.batch * 2, DType::I32),
            TType::new(self.batch * LEGAL_SLOTS, DType::I32),
        ]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let losses = self.run_kernel("policy_loss", &inputs, TType::new(self.batch, DType::F32), false)?;
        Ok(vec![losses.reduce_sum([self.batch, 1], 0)?])
    }

    fn backward<'a>(&self, mut inputs: Vec<TNode<'a>>, grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        inputs.push(grads[0]);
        Ok(vec![self.run_kernel(
            "policy_loss_bwd",
            &inputs,
            inputs[0].ty(),
            true,
        )?])
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        other.downcast::<Self>().is_some_and(|other| self == other)
    }
}

#[derive(Clone, Copy, Debug)]
struct LegalLoss;

impl ModelOperation for LegalLoss {
    fn opname(&self) -> String {
        "LegalLoss".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![
            MType::new(true, MOVES, 1, Layout::Dense(DType::F32)),
            MType::new(true, LEGAL_SLOTS, 1, Layout::Dense(DType::F32)),
            MType::new(true, LEGAL_SLOTS, 1, Layout::Sparse(2)),
            MType::new(true, MOVES, 1, Layout::Sparse(LEGAL_SLOTS)),
        ]
    }

    fn output(&self) -> MType {
        MType::new(false, 1, 1, Layout::Dense(DType::F32))
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        lower
            .add_op(inputs, CustomAutogradOp::new(LegalLossAutograd { batch: batch_size }))
            .map(|x| x[0])
    }
}

fn filter(_pos: &Position, _mv: Move, score: Score, _result: GameResult) -> bool {
    !is_decisive(score) && rng().random_bool(0.5)
}

fn load_ft(path: &str) -> (Vec<f32>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    let inputs = TakTopRelativeStacks::NUM_INPUTS;
    let values = bytes[..(inputs + 1) * L1 * 2].as_chunks::<2>().0;
    let unpack = |values: &[[u8; 2]]| {
        values
            .as_chunks::<L1>()
            .0
            .iter()
            .flat_map(|row| {
                (0..L1).map(|neuron| f32::from(i16::from_le_bytes(row[packus_position(neuron)])) / Q0 as f32)
            })
            .collect()
    };
    (unpack(&values[..inputs * L1]), unpack(&values[inputs * L1..]))
}

fn fill_chunk(
    data: &[LoadedTakBoard],
    move_offset: usize,
    stm: &mut [i32],
    ntm: &mut [i32],
    ranges: &mut [i32],
    moves: &mut [i32],
    targets: &mut [f32],
) {
    let nnz = TakTopRelativeStacks.max_active();
    let mut legal = Vec::new();
    let mut used = 0;

    for (i, board) in data.iter().enumerate() {
        let pos = board.decode_position().unwrap();
        generate_moves(&mut legal, &pos);
        if used + legal.len() > moves.len() {
            stm[i * nnz..].fill(-1);
            ntm[i * nnz..].fill(-1);
            ranges[2 * i..].fill(0);
            break;
        }

        let stm = &mut stm[i * nnz..(i + 1) * nnz];
        let ntm = &mut ntm[i * nnz..(i + 1) * nnz];
        let mut count = 0;
        TakTopRelativeStacks.map_position(&pos, |our, opp| {
            stm[count] = our as i32;
            ntm[count] = opp as i32;
            count += 1;
        });
        if count < nnz {
            stm[count] = -1;
            ntm[count] = -1;
        }

        let transform = normalizer(pos.cap_sq(pos.stm()));
        ranges[2 * i] = (move_offset + used) as i32;
        ranges[2 * i + 1] = legal.len() as i32;
        for (j, &mv) in legal.iter().enumerate() {
            moves[used + j] = move_index(mv, transform) as i32;
            targets[used + j] = if mv == board.2 { 1.0 } else { 0.0 };
        }
        used += legal.len();
    }
}

struct PolicyLoader {
    reader: TakReader,
}

impl PolicyLoader {
    fn prepare<G: Gpu>(
        &self,
        pool: &Arc<HostPool<G>>,
        data: &[LoadedTakBoard],
    ) -> Result<PreparedBatchHost<G>, G::Error> {
        let batch = data.len();
        let nnz = TakTopRelativeStacks.max_active();
        let mut stm = pool.take(DType::I32, batch * nnz)?;
        let mut ntm = pool.take(DType::I32, batch * nnz)?;
        let mut ranges = pool.take(DType::I32, batch * 2)?;
        let mut moves = pool.take(DType::I32, batch * LEGAL_SLOTS)?;
        let mut targets = pool.take(DType::F32, batch * LEGAL_SLOTS)?;

        let size = batch.div_ceil(MAP_THREADS);
        std::thread::scope(|s| {
            let mut stm = stm.as_mut_slice::<i32>().unwrap().chunks_mut(size * nnz);
            let mut ntm = ntm.as_mut_slice::<i32>().unwrap().chunks_mut(size * nnz);
            let mut ranges = ranges.as_mut_slice::<i32>().unwrap().chunks_mut(size * 2);
            let mut moves = moves.as_mut_slice::<i32>().unwrap().chunks_mut(size * LEGAL_SLOTS);
            let mut targets = targets.as_mut_slice::<f32>().unwrap().chunks_mut(size * LEGAL_SLOTS);
            for (i, data) in data.chunks(size).enumerate() {
                let move_offset = i * size * LEGAL_SLOTS;
                let (stm, ntm) = (stm.next().unwrap(), ntm.next().unwrap());
                let (ranges, moves) = (ranges.next().unwrap(), moves.next().unwrap());
                let targets = targets.next().unwrap();
                s.spawn(move || fill_chunk(data, move_offset, stm, ntm, ranges, moves, targets));
            }
        });

        let inputs = [
            ("stm", stm),
            ("ntm", ntm),
            ("ranges", ranges),
            ("moves", moves),
            ("targets", targets),
        ]
        .map(|(name, buffer)| (name.to_string(), buffer))
        .into_iter()
        .collect();
        Ok(PreparedBatchHost::new(pool.clone(), inputs))
    }
}

impl DataLoader for PolicyLoader {
    fn map_batches<G: Gpu, F: FnMut(PreparedBatchHost<G>) -> bool>(
        self,
        pool: &Arc<HostPool<G>>,
        _start: Step,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError> {
        let mut result = Ok(());
        self.reader.read_chunks(0, |chunk| {
            chunk
                .chunks_exact(batch_size)
                .any(|data| match self.prepare(pool, data) {
                    Ok(batch) => f(batch),
                    Err(e) => {
                        result = Err(DataLoadingError::Message(format!("{e:?}")));
                        true
                    }
                })
        });
        result
    }
}

fn main() {
    let mut id = "policy".to_owned();
    let mut net = String::new();
    let mut superbatches = 50;
    let mut data = vec![];

    let mut iter = std::env::args().skip(1).peekable();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--id" => id = iter.next().unwrap(),
            "--net" => net = iter.next().unwrap(),
            "--superbatches" => superbatches = iter.next().unwrap().parse().unwrap(),
            "--data" => {
                while let Some(file) = iter.next_if(|arg| !arg.starts_with("--")) {
                    data.push(file);
                }
            }
            _ => panic!("unknown {flag}"),
        }
    }

    assert!(!net.is_empty());
    assert!(!data.is_empty());

    std::fs::create_dir_all("checkpoints").unwrap();

    let num_inputs = TakTopRelativeStacks::NUM_INPUTS;
    let nnz = TakTopRelativeStacks.max_active();

    let inputs = ModelInputs::default()
        .add_sparse("stm", (num_inputs, 1), nnz)
        .add_sparse("ntm", (num_inputs, 1), nnz)
        .add_sparse("ranges", (LEGAL_SLOTS, 1), 2)
        .add_sparse("moves", (MOVES, 1), LEGAL_SLOTS)
        .add_dense("targets", (LEGAL_SLOTS, 1));

    let builder = ModelBuilder::default();
    let ((((stm, ntm), ranges), moves), targets) = inputs.make_nodes(&builder);

    let l0 = builder
        .new_affine("l0", num_inputs, L1)
        .with_precision(MatmulPrecision::Half);
    let p = builder.new_affine("p", L1, MOVES).with_precision(MatmulPrecision::Half);

    let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
    let stm_hidden = ft(stm, 0, L1 / 2) * ft(stm, L1 / 2, L1);
    let ntm_hidden = ft(ntm, 0, L1 / 2) * ft(ntm, L1 / 2, L1);
    let activated = stm_hidden.concat(ntm_hidden).detach();

    let logits = p.forward(activated);
    let loss = builder.add_op([logits, targets, ranges, moves], LegalLoss);
    let defn = ModelDefinition::new(builder.inner(), Some(loss), vec![(logits.node(), "output".to_string())]);

    let (l0w, l0b) = load_ft(&net);
    let mut weights = ModelWeights::new(&defn, 1);
    assert!(weights.set("l0w", TValue::F32(l0w)));
    assert!(weights.set("l0b", TValue::F32(l0b)));

    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams {
        decay: 0.0,
        min_weight: f32::MIN,
        max_weight: f32::MAX,
        ..Default::default()
    };
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device, params).unwrap();
    let weight_params = AdamWParams {
        min_weight: i8::MIN as f32 / POLICY_WEIGHT_SCALE,
        max_weight: i8::MAX as f32 / POLICY_WEIGHT_SCALE,
        ..params
    };
    optimiser.set_params_for_weight("pw", weight_params);
    let bias_params = AdamWParams {
        min_weight: i32::MIN as f32 / POLICY_DIVISOR,
        max_weight: (i32::MAX as f32).next_down() / POLICY_DIVISOR,
        ..params
    };
    optimiser.set_params_for_weight("pb", bias_params);

    let saved_format = vec![
        SavedFormat::id("pw")
            .transpose()
            .transform(|_, values| values.iter().map(|w| w * POLICY_WEIGHT_SCALE).collect())
            .round()
            .quantise::<i8>(1),
        SavedFormat::id("pb").round().quantise::<i32>(POLICY_DIVISOR as i32),
    ];

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        lr_schedule: lr::CosineDecayLR {
            initial_lr: 1e-3,
            final_lr: 1e-5,
            final_superbatch: superbatches,
        }
        .boxed(),
        log_rate: 128,
    };

    train(
        &mut optimiser,
        schedule,
        PolicyLoader {
            reader: TakReader::new(data, SHUFFLE_BUFFER, LOAD_THREADS, filter),
        },
        |_, _, _| {},
        |optimiser, step| {
            if step.superbatch() != step.final_superbatch() {
                return;
            }
            let name = format!("{id}-{}", step.superbatch());
            save_to_checkpoint(optimiser, &saved_format, &format!("checkpoints/{name}"));
            println!("Saved [{name}]");
        },
    )
    .unwrap();
}
