use crate::board::Position;
use crate::core::PieceType;
use crate::takmove::Move;

fn generate_starting_moves(dst: &mut Vec<Move>, pos: &Position) {
    for sq in !pos.occ() {
        dst.push(Move::placement(PieceType::Flat, sq));
    }
}

pub fn generate_moves(dst: &mut Vec<Move>, pos: &Position) {
    dst.clear();

    if pos.ply() < 2 {
        generate_starting_moves(dst, pos);
        return;
    }

    todo!();
}
