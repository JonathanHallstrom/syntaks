use crate::board::Position;
use crate::core::{Direction, PieceType, Square};
use crate::takmove::Move;

fn generate_starting_moves(dst: &mut Vec<Move>, pos: &Position) {
    for sq in !pos.occ() {
        dst.push(Move::placement(PieceType::Flat, sq));
    }
}

fn generate_placements(dst: &mut Vec<Move>, pos: &Position) {
    let flats = pos.flats_in_hand(pos.stm());
    let caps = pos.caps_in_hand(pos.stm());

    if flats == 0 && caps == 0 {
        return;
    }

    for sq in !pos.occ() {
        if caps > 0 {
            dst.push(Move::placement(PieceType::Capstone, sq));
        }

        if flats > 0 {
            dst.push(Move::placement(PieceType::Flat, sq));
            dst.push(Move::placement(PieceType::Wall, sq));
        }
    }
}

#[must_use]
fn find_hit(pos: &Position, start: Square, dir: Direction) -> (u32, Option<PieceType>) {
    //TODO non-toy impl

    let mut sq = start;
    let mut dist = 0;
    let mut top = None;

    while let Some(next) = sq.shift_checked(dir) {
        sq = next;
        top = pos.stacks().top(sq);
        dist += 1;

        if let Some(top_pt) = top
            && matches!(top_pt, PieceType::Capstone | PieceType::Wall)
        {
            break;
        }
    }

    (dist, top)
}

fn do_spreads(
    dst: &mut Vec<Move>,
    sq: Square,
    dir: Direction,
    lsb: u16,
    mut pattern: u16,
    dist: u32,
    limit: u16,
) {
    assert!(dist > 0);
    while pattern < limit {
        dst.push(Move::spread(sq, dir, pattern));
        if pattern.count_ones() == dist {
            pattern += pattern & pattern.wrapping_neg();
        } else {
            pattern += lsb;
        }
    }
}

fn generate_spreads(dst: &mut Vec<Move>, pos: &Position) {
    for sq in pos.player_bb(pos.stm()) {
        let top = pos.stacks().top(sq).unwrap();
        let max = pos.stacks().height(sq).min(6);

        let start_bit = (Move::PATTERN_MASK + 1) >> max;

        for dir in [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ] {
            let (mut dist, hit_top) = find_hit(pos, sq, dir);

            if dist == 0 {
                continue;
            }

            let mut limit = Move::PATTERN_MASK + 1;

            match hit_top {
                Some(PieceType::Wall) => {
                    if top == PieceType::Capstone {
                        // Smashes
                        do_spreads(dst, sq, dir, start_bit, Move::PATTERN_MASK, dist, limit);
                        limit -= 1;
                    }
                    dist -= 1;
                }
                Some(PieceType::Capstone) => {
                    dist -= 1;
                }
                _ => {}
            }

            if dist == 0 {
                continue;
            }

            do_spreads(dst, sq, dir, start_bit, start_bit, dist, limit);
        }
    }
}

pub fn generate_moves(dst: &mut Vec<Move>, pos: &Position) {
    dst.clear();

    if pos.ply() < 2 {
        generate_starting_moves(dst, pos);
        return;
    }

    generate_placements(dst, pos);
    generate_spreads(dst, pos);
}
