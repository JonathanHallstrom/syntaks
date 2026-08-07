use crate::board::{Position, Stacks};
use crate::core::{PieceType, Player, Square};
use crate::takmove::Move;
use std::io::{Read, Write};

pub const POSITION_RECORD_SIZE: usize = 32;

const BOARD_BITS: usize = 200;
const FLAGS_BYTE: usize = 25;
const RESERVES_BYTES: usize = 26;
const COUNT_BYTES: usize = 28;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum GameResult {
    Loss,
    Draw,
    Win,
}

impl GameResult {
    #[must_use]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Loss),
            1 => Some(Self::Draw),
            2 => Some(Self::Win),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::Draw => Self::Draw,
            Self::Win => Self::Loss,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum DecodeError {
    Overrun,
    InvalidPosition,
    ReserveMismatch,
}

fn put_tokem(buf: &mut [u8], bit: &mut usize, token: u8) {
    debug_assert!(token < 4);
    buf[*bit >> 3] |= token << (*bit & 7);
    *bit += 2;
}

fn get_token(buf: &[u8], bit: &mut usize) -> u8 {
    let token = (buf[*bit >> 3] >> (*bit & 7)) & 0b11;
    *bit += 2;
    token
}

#[must_use]
pub fn encode_position(pos: &Position) -> [u8; POSITION_RECORD_SIZE] {
    let mut out = [0u8; POSITION_RECORD_SIZE];
    let mut bit = 0;

    let stacks = pos.stacks();

    for sq in Square::all() {
        match stacks.top(sq) {
            None => put_tokem(&mut out, &mut bit, 0b00),
            Some(pt) => {
                put_tokem(&mut out, &mut bit, pt.raw() + 1);

                let height = stacks.height(sq);
                let players = stacks.players(sq);

                for level in (0..height).rev() {
                    let color = ((players >> level) & 1) as u8;
                    let terminal = u8::from(level == 0);
                    put_tokem(&mut out, &mut bit, (terminal << 1) | color);
                }
            }
        }
    }

    debug_assert!(bit <= BOARD_BITS);

    out[FLAGS_BYTE] = pos.stm().raw() | (u8::from(pos.ply() < 2) << 1);
    out[RESERVES_BYTES] = pos.flats_in_hand(Player::P1);
    out[RESERVES_BYTES + 1] = pos.flats_in_hand(Player::P2);

    out
}

pub fn decode_position(rec: &[u8; POSITION_RECORD_SIZE]) -> Result<Position, DecodeError> {
    let mut bit = 0;

    let mut pos = Position::startpos();

    let mut flats_used = [0u32; 2];
    let mut caps_used = [0u32; 2];

    let mut colors = [Player::P1; Stacks::MAX_HEIGHT];

    for sq in Square::all() {
        let header = get_token(rec, &mut bit);
        if header == 0 {
            continue;
        }

        let top = PieceType::from_raw(header - 1).unwrap();

        let mut height = 0;
        loop {
            if bit + 2 > BOARD_BITS || height >= Stacks::MAX_HEIGHT {
                return Err(DecodeError::Overrun);
            }

            let token = get_token(rec, &mut bit);
            let player = Player::from_raw(token & 1).unwrap();

            if height == 0 && top == PieceType::Capstone {
                caps_used[player.idx()] += 1;
            } else {
                flats_used[player.idx()] += 1;
            }
            // println!("{height}");

            colors[height] = player;
            height += 1;

            if token & 0b10 != 0 {
                break;
            }
        }

        for idx in (0..height).rev() {
            let pt = if idx == 0 { top } else { PieceType::Flat };
            pos.push_piece(sq, pt, colors[idx]);
        }
    }

    if flats_used.iter().any(|&used| used > 30) || caps_used.iter().any(|&used| used > 1) {
        return Err(DecodeError::InvalidPosition);
    }

    let stm = Player::from_raw(rec[FLAGS_BYTE] & 1).unwrap();
    let swap_phase = rec[FLAGS_BYTE] & 0b10 != 0;

    pos.finish_build(stm, u16::from(stm.raw()) + if swap_phase { 0 } else { 2 });

    if pos.flats_in_hand(Player::P1) != rec[RESERVES_BYTES] || pos.flats_in_hand(Player::P2) != rec[RESERVES_BYTES + 1]
    {
        return Err(DecodeError::ReserveMismatch);
    }

    Ok(pos)
}

#[derive(Clone, Debug)]
pub struct Game {
    pub root: Position,
    pub result: GameResult,
    pub moves: Vec<(Move, i16)>,
}

pub fn write_game(
    writer: &mut impl Write,
    root: &Position,
    result: GameResult,
    moves: &[(Move, i16)],
) -> std::io::Result<()> {
    assert!(moves.len() <= u16::MAX as usize);

    let mut buf = Vec::with_capacity(POSITION_RECORD_SIZE + 4 * moves.len());

    buf.extend_from_slice(&encode_position(root));
    buf[FLAGS_BYTE] |= result.raw() << 2;
    buf[COUNT_BYTES..COUNT_BYTES + 2].copy_from_slice(&(moves.len() as u16).to_le_bytes());

    for &(mv, eval) in moves {
        buf.extend_from_slice(&mv.raw().to_le_bytes());
        buf.extend_from_slice(&eval.to_le_bytes());
    }

    writer.write_all(&buf)
}

fn invalid_data(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_owned())
}

pub fn read_game(reader: &mut impl Read) -> std::io::Result<Option<Game>> {
    let mut rec = [0u8; POSITION_RECORD_SIZE];

    if reader.read(&mut rec[..1])? == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut rec[1..])?;

    let result = GameResult::from_raw((rec[FLAGS_BYTE] >> 2) & 0b11).ok_or_else(|| invalid_data("bad game result"))?;
    let count = u16::from_le_bytes([rec[COUNT_BYTES], rec[COUNT_BYTES + 1]]) as usize;

    let root = decode_position(&rec).map_err(|err| invalid_data(&format!("bad position: {:?}", err)))?;

    let mut buf = vec![0u8; 4 * count];
    reader.read_exact(&mut buf)?;

    let moves = buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| {
            let mv = Move::from_raw(u16::from_le_bytes([chunk[0], chunk[1]]))?;
            Some((mv, i16::from_le_bytes([chunk[2], chunk[3]])))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid_data("null move"))?;

    Ok(Some(Game { root, result, moves }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_moves;
    use crate::prng::Sfc64;

    fn assert_round_trip(pos: &Position) {
        let enc = encode_position(pos);
        let dec = decode_position(&enc).unwrap();

        assert_eq!(dec.key(), pos.key(), "key mismatch for {}", pos.tps());
        assert_eq!(dec.flats_in_hand(Player::P1), pos.flats_in_hand(Player::P1));
        assert_eq!(dec.flats_in_hand(Player::P2), pos.flats_in_hand(Player::P2));
        assert_eq!(dec.caps_in_hand(Player::P1), pos.caps_in_hand(Player::P1));
        assert_eq!(dec.caps_in_hand(Player::P2), dec.caps_in_hand(Player::P2));
        assert_eq!(dec.stm(), pos.stm());
        assert_eq!(encode_position(&dec), enc);
    }

    fn is_terminal(pos: &Position) -> bool {
        pos.has_road(Player::P1)
            || pos.has_road(Player::P2)
            || !matches!(pos.count_flats(), crate::board::FlatCountOutcome::None)
    }

    #[test]
    fn position_round_trip() {
        let mut rng = Sfc64::new(0xdead_beef);
        let mut moves = Vec::new();

        assert_round_trip(&Position::startpos());

        for _ in 0..200 {
            let mut pos = Position::startpos();

            for _ in 0..400 {
                assert_round_trip(&pos);

                moves.clear();
                generate_moves(&mut moves, &pos);

                let mv = moves[rng.next_u64() as usize % moves.len()];
                pos = pos.apply_move(mv);

                if is_terminal(&pos) {
                    break;
                }
            }

            assert_round_trip(&pos);
        }
    }
}
