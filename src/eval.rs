/*
 * syntaks, a TEI Tak engine
 * Copyright (c) 2026 Ciekce
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use crate::board::Position;
use crate::core::{PieceType, Player, Square};
use crate::search::{SCORE_WIN, Score};

const HL: usize = 32;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

#[repr(C, align(64))]
struct Network {
    l0w: [[i16; HL]; 216],
    l0b: [i16; HL],
    l1w: [[[i16; HL]; 2]; 2],
    l1b: [i16; 2],
}

static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("EVALFILE"))) };

fn feature(perspective: Player, side: Player, pt: PieceType, sq: Square) -> usize {
    // TODO: was tired and got side and piecetype backwards
    pt.idx() * 72 + usize::from(side != perspective) * 36 + sq.idx()
}

#[must_use]
pub fn static_eval(pos: &Position) -> Score {
    let stm = pos.stm();
    let stacks = pos.stacks();

    let mut accs = [NET.l0b; 2];

    for side in [Player::P1, Player::P2] {
        for sq in pos.player_bb(side) {
            let pt = stacks.top(sq).unwrap();

            for perspective in [Player::P1, Player::P2] {
                for (a, w) in accs[perspective.idx()]
                    .iter_mut()
                    .zip(&NET.l0w[feature(perspective, side, pt, sq)])
                {
                    *a += w;
                }
            }
        }
    }

    let mut sum = 0;

    for (perspective, weights) in [stm, stm.flip()].iter().zip(&NET.l1w[stm.idx()]) {
        for (&a, &w) in accs[perspective.idx()].iter().zip(weights) {
            let c = a.clamp(0, QA as i16);
            sum += i32::from(c) * i32::from(c * w);
        }
    }

    let eval = (sum / QA + i32::from(NET.l1b[stm.idx()])) * SCALE / (QA * QB);

    (eval as Score).clamp(-SCORE_WIN, SCORE_WIN)
}
