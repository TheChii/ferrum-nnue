//! Integration test for Aurora NNUE (768→256)×2→1 architecture

use nnue::*;
use nnue::aurora::{load_model, AuroraModel};

/// Parse a FEN string and return pieces list and side to move
fn parse_fen(fen: &str) -> (Vec<(Piece, Color, Square)>, Color) {
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
    
    (pieces, side_to_move)
}

fn activate(model: &AuroraModel, fen: &str) -> i32 {
    let (pieces, side_to_move) = parse_fen(fen);
    let mut state = model.new_state();
    
    // Add all pieces to the state
    for &(piece, piece_color, square) in &pieces {
        state.add(piece, piece_color, square);
    }
    
    state.activate(side_to_move)
}

#[test]
fn aurora_loads_and_evaluates() {
    // Load the Aurora model
    let model = load_model("andromeda-3.nnue").expect("Failed to load andromeda-3.nnue");
    
    // Test on starting position
    let start_eval = activate(&model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    println!("Starting position eval: {}", start_eval);
    
    // Evaluation should be close to 0 for starting position (within ±100 cp)
    assert!(start_eval.abs() < 100, "Starting position eval {} seems wrong", start_eval);
}

#[test]
fn aurora_evaluates_positions() {
    let model = load_model("andromeda-3.nnue").expect("Failed to load andromeda-3.nnue");
    
    // Test that NNUE produces reasonable evaluations
    let start = activate(&model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let white_up_queen = activate(&model, "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let black_up_queen = activate(&model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    
    println!("Start position: {}", start);
    println!("White up queen: {}", white_up_queen);
    println!("Black up queen: {}", black_up_queen);
    
    // White being up a queen should evaluate better than start position
    assert!(white_up_queen > start, "White up queen ({}) should be > start ({})", white_up_queen, start);
    // Black being up a queen should evaluate worse than start position  
    assert!(black_up_queen < start, "Black up queen ({}) should be < start ({})", black_up_queen, start);
    // Relative ordering should be correct
    assert!(white_up_queen > black_up_queen, "White up ({}) should be > black up ({})", white_up_queen, black_up_queen);
}

#[test]
fn aurora_perspective_symmetry() {
    let model = load_model("andromeda-3.nnue").expect("Failed to load andromeda-3.nnue");
    
    // These two positions are mirrors of each other
    // White to move with advantage vs Black to move with same advantage (colors swapped)
    let white_up = activate(&model, "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    let black_up = activate(&model, "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    
    println!("White perspective: {}, Black perspective: {}", white_up, black_up);
    
    // Both should be similar (slight advantage for the side that just moved Nf3/Nf6)
    // The absolute difference should be small since positions are symmetric
    assert!((white_up - black_up).abs() < 50, "Perspective symmetry broken: {} vs {}", white_up, black_up);
}
