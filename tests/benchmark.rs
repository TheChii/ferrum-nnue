use std::io::Cursor;
use std::time::Instant;

use binread::BinRead;
use nnue::*;
use nnue::stockfish::halfkp::{SfHalfKpFullModel, SfHalfKpModel, SfHalfKpState};

/// Parse a FEN string and return (pieces, side_to_move, white_king, black_king)
fn parse_fen(fen: &str) -> (Vec<(Piece, Color, Square)>, Color, Square, Square) {
    let mut parts = fen.split_ascii_whitespace();
    let pos = parts.next().unwrap();
    let mut pieces = Vec::new();
    
    for (rank, row) in pos.rsplit("/").enumerate() {
        let mut file = 0;
        for p in row.chars() {
            if let Some(offset) = p.to_digit(10) {
                file += offset as usize;
            } else {
                let piece = match p.to_ascii_lowercase() {
                    'p' => Piece::Pawn,
                    'n' => Piece::Knight,
                    'b' => Piece::Bishop,
                    'r' => Piece::Rook,
                    'q' => Piece::Queen,
                    'k' => Piece::King,
                    _ => panic!("Invalid piece {}", p)
                };
                let color = if p.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let square = Square::from_index(rank * 8 + file);
                pieces.push((piece, color, square));
                file += 1;
            }
        }
    }
    
    let side_to_move = if parts.next().unwrap() == "w" {
        Color::White
    } else {
        Color::Black
    };
    
    let mut white_king = Square::A1;
    let mut black_king = Square::A1;
    for &(piece, color, square) in &pieces {
        if piece == Piece::King {
            if color == Color::White {
                white_king = square;
            } else {
                black_king = square;
            }
        }
    }
    
    (pieces, side_to_move, white_king, black_king)
}

fn setup_state<'a>(model: &'a SfHalfKpModel, fen: &str) -> (SfHalfKpState<'a>, Color) {
    let (pieces, side_to_move, white_king, black_king) = parse_fen(fen);
    let mut state = model.new_state(white_king, black_king);
    
    for &(piece, piece_color, square) in &pieces {
        if piece != Piece::King {
            for &color in &Color::ALL {
                state.add(color, piece, piece_color, square);
            }
        }
    }
    
    (state, side_to_move)
}

const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/p2nppbp/2p3pB/1p2P3/2pP4/2P2N2/P2QBPPP/R4RK1 b - - 7 12",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/5pk1/2R4p/6pP/6P1/pBP5/Pr5r/5K2 w - - 2 48",
];

#[test]
#[ignore] // Run with: cargo test benchmark_nnue_speed --release -- --ignored --nocapture
fn benchmark_nnue_speed() {
    // Load model
    println!("\nLoading NNUE model...");
    let mut reader = Cursor::new(std::fs::read("tests/nn.nnue").unwrap());
    let model = SfHalfKpFullModel::read(&mut reader).unwrap();
    println!("Model loaded: {}", &model.desc[..60]);
    
    // Get iteration count from env or use default
    let iterations: usize = std::env::var("BENCH_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    
    println!("\n{}", "=".repeat(60));
    println!("NNUE INFERENCE BENCHMARK");
    println!("{}", "=".repeat(60));
    println!("Iterations per position: {}", iterations);
    println!("Total positions: {}", BENCH_FENS.len());
    println!("{}", "-".repeat(60));
    
    let mut total_evals = 0u64;
    let mut total_time_ns = 0u128;
    
    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let (mut state, side_to_move) = setup_state(&model.model, fen);
        
        // Warmup
        for _ in 0..100 {
            let _ = state.activate(side_to_move);
        }
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = state.activate(side_to_move);
        }
        let elapsed = start.elapsed();
        
        let elapsed_ns = elapsed.as_nanos();
        let evals_per_sec = (iterations as f64 / elapsed.as_secs_f64()) as u64;
        let ns_per_eval = elapsed_ns / iterations as u128;
        
        total_evals += iterations as u64;
        total_time_ns += elapsed_ns;
        
        println!(
            "[{}/{}] {:>12} evals/sec | {:>6} ns/eval | {}",
            i + 1,
            BENCH_FENS.len(),
            format_number(evals_per_sec),
            ns_per_eval,
            &fen[..40.min(fen.len())]
        );
    }
    
    println!("{}", "-".repeat(60));
    
    let total_secs = total_time_ns as f64 / 1_000_000_000.0;
    let avg_evals_per_sec = (total_evals as f64 / total_secs) as u64;
    let avg_ns_per_eval = total_time_ns / total_evals as u128;
    
    println!("SUMMARY:");
    println!("  Total evaluations:  {}", format_number(total_evals));
    println!("  Total time:         {:.3} seconds", total_secs);
    println!("  Average speed:      {} evals/sec", format_number(avg_evals_per_sec));
    println!("  Average latency:    {} ns/eval", avg_ns_per_eval);
    println!("{}", "=".repeat(60));
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
