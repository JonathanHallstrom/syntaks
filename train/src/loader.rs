use crate::inputs::TakBoard;
use bullet_trainer::reader::DataReader;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc;
use syntaks::format::{Game, read_game};

const SHUFFLE_BUFFER: usize = 1 << 22;

fn splat(game: &Game, out: &mut Vec<TakBoard>) {
    let mut pos = game.root;

    for &(mv, eval) in &game.moves {
        out.push(TakBoard::new(&pos, eval, game.result));
        pos = pos.apply_move(mv);
    }
}

#[derive(Clone)]
pub struct TakReader {
    pub paths: Vec<String>,
}

impl DataReader<TakBoard> for TakReader {
    fn read_chunks<F: FnMut(&[TakBoard]) -> bool>(&self, _skip: usize, mut f: F) {
        let paths = self.paths.clone();
        let (sender, receiver) = mpsc::sync_channel::<Vec<TakBoard>>(2);

        std::thread::spawn(move || {
            let mut buffer: Vec<TakBoard> = Vec::with_capacity(SHUFFLE_BUFFER);

            loop {
                for path in &paths {
                    let mut reader = BufReader::new(File::open(path).unwrap());

                    while let Ok(Some(game)) = read_game(&mut reader) {
                        splat(&game, &mut buffer);

                        if buffer.len() >= SHUFFLE_BUFFER {
                            buffer.shuffle(&mut rand::rng());
                            if sender.send(buffer).is_err() {
                                return;
                            }
                            buffer = Vec::with_capacity(SHUFFLE_BUFFER);
                        }
                    }
                }
            }
        });

        while let Ok(chunk) = receiver.recv() {
            if f(&chunk) {
                break;
            }
        }
    }
}
