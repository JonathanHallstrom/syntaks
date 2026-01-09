use crate::board::Position;
use crate::movegen::generate_moves;
use crate::perft::{perft, split_perft};

const NAME: &str = "syntaks";
const AUTHORS: &str = "Ciekce";
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct TeiHandler {
    pos: Position,
}

impl TeiHandler {
    #[must_use]
    fn new() -> Self {
        Self {
            pos: Position::startpos(),
        }
    }

    fn run(&mut self) {
        let mut line = String::with_capacity(256);
        while let Ok(bytes) = std::io::stdin().read_line(&mut line) {
            if bytes == 0 {
                break;
            }

            let args: Vec<_> = line.split_ascii_whitespace().collect();
            if args.is_empty() {
                line.clear();
                continue;
            }

            let (&command, args) = args.split_first().unwrap();

            match command {
                "tei" => self.handle_tei(),
                "teinewgame" => self.handle_teinewgame(args),
                "setoption" => self.handle_setoption(args),
                "isready" => self.handle_isready(),
                "position" => self.handle_position(args),
                "go" => self.handle_go(args),
                "d" => self.handle_d(),
                "perft" => self.handle_perft(args),
                "splitperft" => self.handle_splitperft(args),
                "quit" => break,
                unknown => eprintln!("Unknown command '{}'", unknown),
            }

            line.clear();
        }
    }

    fn handle_tei(&self) {
        println!("id name {} {}", NAME, VERSION);
        println!("id author {}", AUTHORS);
        println!("option name HalfKomi type spin default 4 min 4 max 4");
        println!("teiok");
    }

    fn handle_teinewgame(&mut self, args: &[&str]) {
        if args.is_empty() {
            println!("info string Missing size, assuming 6x6");
        } else {
            match args[0].parse::<u32>() {
                Ok(size) => {
                    if size != 6 {
                        eprintln!("Only 6x6 supported");
                        return;
                    }
                }
                Err(_) => eprintln!("Invalid size"),
            }
        }

        //NOOP
    }

    fn handle_setoption(&mut self, _args: &[&str]) {
        //NOOP
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_position(&mut self, args: &[&str]) {
        if args.is_empty() {
            return;
        }

        let (&pos_type, args) = args.split_first().unwrap();

        let mut next = 0;

        match pos_type {
            "startpos" => self.pos = Position::startpos(),
            "tps" => {
                let count = args
                    .iter()
                    .position(|&s| s == "moves")
                    .unwrap_or(args.len());

                if count == 0 {
                    eprintln!("Missing TPS");
                    return;
                }

                match Position::from_tps_parts(&args[0..count]) {
                    Ok(pos) => self.pos = pos,
                    Err(err) => {
                        eprintln!("Failed to parse TPS: {:?}", err);
                        return;
                    }
                }

                next += count;
            }
            _ => {
                eprintln!("Invalid position type {}", pos_type);
                return;
            }
        }

        if next >= args.len() || args[next] != "moves" {
            return;
        }

        for &move_str in &args[(next + 1)..] {
            match move_str.parse() {
                Ok(mv) => self.pos = self.pos.apply_move(mv),
                Err(err) => {
                    eprintln!("Invalid move '{}': {:?}", move_str, err);
                    return;
                }
            }
        }
    }

    fn handle_go(&self, _args: &[&str]) {
        let mut moves = Vec::with_capacity(256);
        generate_moves(&mut moves, &self.pos);

        let mv = moves[fastrand::usize(0..moves.len())];

        println!("info depth 1 seldepth 1 nodes 1 score cp 0 pv {}", mv);
        println!("bestmove {}", mv);
    }

    fn handle_d(&self) {
        println!("TPS: {}", self.pos.tps());
        println!("Key: {:016x}", self.pos.key());
    }

    fn handle_perft(&self, args: &[&str]) {
        if args.is_empty() {
            eprintln!("Missing depth");
        }

        let depth = match args[0].parse() {
            Ok(depth) => depth,
            Err(_) => {
                eprintln!("Invalid depth '{}'", args[0]);
                return;
            }
        };

        println!("{}", perft(&self.pos, depth));
    }

    fn handle_splitperft(&self, args: &[&str]) {
        if args.is_empty() {
            eprintln!("Missing depth");
        }

        let depth = match args[0].parse() {
            Ok(depth) => depth,
            Err(_) => {
                eprintln!("Invalid depth '{}'", args[0]);
                return;
            }
        };

        split_perft(&self.pos, depth);
    }
}

pub fn run() {
    let mut handler = TeiHandler::new();
    handler.run();
}
