//! Benchmark comparing Aurora and Stockfish NNUE architectures

use std::io::Cursor;
use std::time::Instant;
use std::hint::black_box;

use binread::BinRead;
use nnue::*;
use nnue::stockfish::halfkp::{SfHalfKpFullModel, SfHalfKpModel, SfHalfKpState};
use nnue::aurora::{load_model as load_aurora, AuroraModel, AuroraState};

/// Parse a FEN string and return pieces list, side to move, and king positions
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

fn setup_stockfish_state<'a>(model: &'a SfHalfKpModel, fen: &str) -> (SfHalfKpState<'a>, Color) {
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

fn setup_aurora_state<'a>(model: &'a AuroraModel, fen: &str) -> (AuroraState<'a>, Color) {
    let (pieces, side_to_move, _, _) = parse_fen(fen);
    let mut state = model.new_state();
    
    for &(piece, piece_color, square) in &pieces {
        state.add(piece, piece_color, square);
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

#[test]
#[ignore]
fn benchmark_aurora_vs_stockfish() {
    let iterations: usize = std::env::var("BENCH_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);

    let mut output = String::new();
    
    output.push_str(&format!("\n{}\n", "=".repeat(70)));
    output.push_str("NNUE ARCHITECTURE COMPARISON BENCHMARK\n");
    output.push_str(&format!("{}\n", "=".repeat(70)));
    output.push_str(&format!("Iterations per position: {}\n", iterations));
    output.push_str(&format!("Total positions: {}\n\n", BENCH_FENS.len()));

    // ===== STOCKFISH BENCHMARK =====
    output.push_str(&format!("{}\n", "-".repeat(70)));
    output.push_str("STOCKFISH HalfKP (40960→256×2→32→32→1) - network.nnue\n");
    output.push_str(&format!("{}\n", "-".repeat(70)));
    
    let sf_model = {
        let data = std::fs::read("network.nnue").expect("Failed to read network.nnue");
        let mut reader = Cursor::new(data);
        SfHalfKpFullModel::read(&mut reader).expect("Failed to parse Stockfish model")
    };
    
    let mut sf_total_evals = 0u64;
    let mut sf_total_time_ns = 0u128;
    
    for fen in BENCH_FENS.iter() {
        let (mut state, side_to_move) = setup_stockfish_state(&sf_model.model, fen);
        
        // Warmup
        for _ in 0..100 {
            let _ = black_box(state.activate(side_to_move));
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(state.activate(side_to_move));
        }
        let elapsed = start.elapsed();
        
        sf_total_evals += iterations as u64;
        sf_total_time_ns += elapsed.as_nanos();
    }
    
    let sf_total_secs = sf_total_time_ns as f64 / 1_000_000_000.0;
    let sf_evals_per_sec = (sf_total_evals as f64 / sf_total_secs) as u64;
    let sf_ns_per_eval = sf_total_time_ns / sf_total_evals as u128;
    
    output.push_str(&format!("  Total evaluations:  {}\n", format_number(sf_total_evals)));
    output.push_str(&format!("  Total time:         {:.3} seconds\n", sf_total_secs));
    output.push_str(&format!("  Speed:              {} evals/sec\n", format_number(sf_evals_per_sec)));
    output.push_str(&format!("  Latency:            {} ns/eval\n\n", sf_ns_per_eval));

    // ===== AURORA BENCHMARK =====
    output.push_str(&format!("{}\n", "-".repeat(70)));
    output.push_str("AURORA (768→256)×2→1 - andromeda-3.nnue\n");
    output.push_str(&format!("{}\n", "-".repeat(70)));
    
    let aurora_model = load_aurora("andromeda-3.nnue").expect("Failed to load Aurora model");
    
    let mut aurora_total_evals = 0u64;
    let mut aurora_total_time_ns = 0u128;
    
    for fen in BENCH_FENS.iter() {
        let (state, side_to_move) = setup_aurora_state(&aurora_model, fen);
        
        // Warmup
        for _ in 0..100 {
            let _ = black_box(state.activate(side_to_move));
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(state.activate(side_to_move));
        }
        let elapsed = start.elapsed();
        
        aurora_total_evals += iterations as u64;
        aurora_total_time_ns += elapsed.as_nanos();
    }
    
    let aurora_total_secs = aurora_total_time_ns as f64 / 1_000_000_000.0;
    let aurora_evals_per_sec = (aurora_total_evals as f64 / aurora_total_secs) as u64;
    let aurora_ns_per_eval = aurora_total_time_ns / aurora_total_evals as u128;
    
    output.push_str(&format!("  Total evaluations:  {}\n", format_number(aurora_total_evals)));
    output.push_str(&format!("  Total time:         {:.3} seconds\n", aurora_total_secs));
    output.push_str(&format!("  Speed:              {} evals/sec\n", format_number(aurora_evals_per_sec)));
    output.push_str(&format!("  Latency:            {} ns/eval\n\n", aurora_ns_per_eval));

    // ===== COMPARISON =====
    output.push_str(&format!("{}\n", "=".repeat(70)));
    output.push_str("COMPARISON\n");
    output.push_str(&format!("{}\n", "=".repeat(70)));
    
    let speedup = aurora_evals_per_sec as f64 / sf_evals_per_sec as f64;
    
    output.push_str(&format!("| Architecture     | Evals/sec      | ns/eval | Relative |\n"));
    output.push_str(&format!("|------------------|----------------|---------|----------|\n"));
    output.push_str(&format!("| Stockfish HalfKP | {:>14} | {:>7} | 1.00x    |\n", 
        format_number(sf_evals_per_sec), sf_ns_per_eval));
    output.push_str(&format!("| Aurora           | {:>14} | {:>7} | {:.2}x    |\n", 
        format_number(aurora_evals_per_sec), aurora_ns_per_eval, speedup));
    output.push_str(&format!("{}\n", "=".repeat(70)));
    
    if speedup > 1.0 {
        output.push_str(&format!("\nAurora is {:.1}x FASTER than Stockfish HalfKP\n", speedup));
    } else {
        output.push_str(&format!("\nStockfish HalfKP is {:.1}x FASTER than Aurora\n", 1.0/speedup));
    }
    
    // Print to console
    println!("{}", output);
    
    // Save to file
    std::fs::write("benchmark_comparison.txt", &output).expect("Failed to write results");
    println!("\nResults saved to benchmark_comparison.txt");
}
