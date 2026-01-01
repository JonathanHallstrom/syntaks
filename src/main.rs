use crate::board::Position;
use crate::perft::split_perft;

mod bitboard;
mod board;
mod core;
mod movegen;
mod perft;
mod takmove;

fn main() {
    let pos = Position::startpos();
    split_perft(&pos, 2);
}
