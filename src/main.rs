use crate::board::Position;
use crate::core::{PieceType, Square};
use crate::movegen::generate_moves;
use crate::perft::split_perft;
use crate::takmove::Move;

mod bitboard;
mod board;
mod core;
mod movegen;
mod perft;
mod takmove;

fn main() {
    let pos = Position::startpos()
        .apply_move(Move::placement(PieceType::Flat, Square::A1))
        .apply_move(Move::placement(PieceType::Flat, Square::B1))
        .apply_move(Move::placement(PieceType::Capstone, Square::F6))
        .apply_move(Move::placement(PieceType::Wall, Square::F1));
    let mut moves = Vec::new();
    generate_moves(&mut moves, &pos);
    //split_perft(&pos, 1);
    println!("{}", pos.tps());
    println!(
        "{}",
        moves
            .iter()
            .filter(|mv| mv.is_spread())
            .map(|mv| mv.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
}
