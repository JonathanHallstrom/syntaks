use crate::board::Position;
use crate::core::{Direction, PieceType, Square};
use crate::perft::split_perft;
use crate::takmove::Move;

mod bitboard;
mod board;
mod core;
mod movegen;
mod perft;
mod takmove;

fn main() {
    let pos = Position::startpos();
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::placement(PieceType::Flat, Square::A1));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::placement(PieceType::Flat, Square::B1));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::spread(Square::B1, Direction::Left, 0b100000));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::placement(PieceType::Flat, Square::A2));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::spread(Square::A1, Direction::Right, 0b100000));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::spread(Square::A1, Direction::Up, 0b100000));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::placement(PieceType::Capstone, Square::A1));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::placement(PieceType::Capstone, Square::C1));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::spread(Square::A1, Direction::Up, 0b100000));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());
    let pos = pos.apply_move(Move::placement(PieceType::Flat, Square::A1));
    assert_eq!(pos, pos.tps().parse::<Position>().unwrap());

    split_perft(&pos, 3);
    println!("{}", pos.tps());
}
