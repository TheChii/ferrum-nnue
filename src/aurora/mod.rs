//! Aurora NNUE: (768→256)×2→1 architecture
//! 
//! Features: 12 pieces × 64 squares = 768 inputs
//! Uses dual perspective accumulators (white/black view) like HalfKP
//! but with simpler feature set for faster evaluation.
//!
//! Binary format (little-endian, from Bullet trainer):
//! - Input weights: 768 × 256 i16 values (393,216 bytes)
//! - Input biases: 256 i16 values (512 bytes)
//! - Output weights: 512 i16 values (1,024 bytes) - for both perspectives
//! - Output bias: 1 i16 value (2 bytes)
//! Total: ~395KB

use std::io::Read;

use crate::layers::BitDense;
use crate::{Square, Piece, Color};

// Network dimensions for Aurora architecture
pub const INPUTS: usize = 768;      // 12 pieces × 64 squares
pub const HIDDEN: usize = 256;      // Hidden layer size
pub const OUTPUTS: usize = 1;       // Single output

/// Aurora NNUE model (768→256)×2→1
/// Uses perspective accumulators like Stockfish HalfKP
#[derive(Debug, Clone)]
pub struct AuroraModel {
    pub input_layer: BitDense<i16, INPUTS, HIDDEN>,
    pub output_weights: [i16; HIDDEN * 2],  // Weights for both perspectives
    pub output_bias: i16,
}

/// Aurora NNUE state with dual perspective accumulators
#[derive(Clone)]
pub struct AuroraState<'a> {
    pub model: &'a AuroraModel,
    /// Accumulators for [white_perspective, black_perspective]
    pub accumulators: [[i16; HIDDEN]; 2],
}

impl AuroraModel {
    /// Create a new state with accumulators initialized to biases
    pub fn new_state(&self) -> AuroraState<'_> {
        let mut accumulators = [[0i16; HIDDEN]; 2];
        self.input_layer.empty(&mut accumulators[0]);
        self.input_layer.empty(&mut accumulators[1]);
        AuroraState {
            model: self,
            accumulators,
        }
    }
}

impl<'a> AuroraState<'a> {
    /// Compute feature index for white's perspective: piece_type * 64 + square
    /// For black's perspective, we flip the square vertically and swap piece colors
    #[inline]
    fn feature_index_white(piece: Piece, piece_color: Color, square: Square) -> usize {
        // piece_type: 0-5 for white pieces, 6-11 for black pieces
        let piece_type = piece_color as usize * 6 + piece as usize;
        piece_type * 64 + square as usize
    }

    /// Compute feature index for black's perspective
    /// Square is flipped, colors are swapped
    #[inline]
    fn feature_index_black(piece: Piece, piece_color: Color, square: Square) -> usize {
        // Swap colors for black's perspective
        let swapped_color = !piece_color;
        let piece_type = swapped_color as usize * 6 + piece as usize;
        // Flip square vertically (rank mirror)
        let flipped_square = square.flip();
        piece_type * 64 + flipped_square as usize
    }

    /// Add a piece to both perspective accumulators
    #[inline]
    pub fn add(&mut self, piece: Piece, piece_color: Color, square: Square) {
        let white_idx = Self::feature_index_white(piece, piece_color, square);
        let black_idx = Self::feature_index_black(piece, piece_color, square);
        self.model.input_layer.add(white_idx, &mut self.accumulators[0]);
        self.model.input_layer.add(black_idx, &mut self.accumulators[1]);
    }

    /// Remove a piece from both perspective accumulators
    #[inline]
    pub fn sub(&mut self, piece: Piece, piece_color: Color, square: Square) {
        let white_idx = Self::feature_index_white(piece, piece_color, square);
        let black_idx = Self::feature_index_black(piece, piece_color, square);
        self.model.input_layer.sub(white_idx, &mut self.accumulators[0]);
        self.model.input_layer.sub(black_idx, &mut self.accumulators[1]);
    }

    /// Refresh accumulators from scratch
    pub fn refresh<F>(&mut self, mut pieces: F)
    where
        F: FnMut() -> Option<(Piece, Color, Square)>,
    {
        self.model.input_layer.empty(&mut self.accumulators[0]);
        self.model.input_layer.empty(&mut self.accumulators[1]);
        while let Some((piece, color, square)) = pieces() {
            self.add(piece, color, square);
        }
    }

    /// Activate the network and get evaluation score
    /// Returns score from the perspective of side_to_move
    #[inline]
    pub fn activate(&self, side_to_move: Color) -> i32 {
        // Select accumulator order based on side to move
        let (stm_acc, opp_acc) = match side_to_move {
            Color::White => (&self.accumulators[0], &self.accumulators[1]),
            Color::Black => (&self.accumulators[1], &self.accumulators[0]),
        };

        // ClippedReLU and output computation
        // Aurora uses clamp to [0, 255] and squared clipped relu
        const QA: i32 = 255;
        
        let mut sum: i32 = 0;
        
        // Process side-to-move accumulator with first half of output weights
        // Aurora: v0 * v0 * weight, accumulated, then divided by 255 at end
        for i in 0..HIDDEN {
            let clamped = (stm_acc[i].max(0).min(255)) as i32;
            let weight = self.model.output_weights[i] as i32;
            // Squared ClippedReLU: clamped * clamped * weight
            sum += clamped * clamped * weight;
        }
        
        // Process opponent accumulator with second half of output weights
        for i in 0..HIDDEN {
            let clamped = (opp_acc[i].max(0).min(255)) as i32;
            let weight = self.model.output_weights[HIDDEN + i] as i32;
            sum += clamped * clamped * weight;
        }
        
        // Divide by QA once, then add bias
        let unsquared = sum / QA + self.model.output_bias as i32;
        
        // Scale to centipawns: (unsquared * 400) / (255 * 64) + 13
        (unsquared * 400) / (QA * 64) + 13
    }
}

/// Scale raw Aurora output to centipawns (already done in activate)
#[inline]
pub fn scale_to_cp(raw: i32) -> i32 {
    raw
}

/// Load an Aurora NNUE model from file
pub fn load_model(path: &str) -> Result<AuroraModel, std::io::Error> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    
    // Read input layer weights: 768 × 256 i16 values
    // Note: Aurora stores as [768][256] (feature-major)
    let mut input_weights = [[0i16; HIDDEN]; INPUTS];
    for i in 0..INPUTS {
        for j in 0..HIDDEN {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            input_weights[i][j] = i16::from_le_bytes(buf);
        }
    }

    // Read input layer biases: 256 i16 values
    let mut input_biases = [0i16; HIDDEN];
    for j in 0..HIDDEN {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        input_biases[j] = i16::from_le_bytes(buf);
    }

    // Read output layer weights: 512 i16 values (256 for each perspective)
    let mut output_weights = [0i16; HIDDEN * 2];
    for i in 0..(HIDDEN * 2) {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        output_weights[i] = i16::from_le_bytes(buf);
    }

    // Read output layer bias: 1 i16 value
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    let output_bias = i16::from_le_bytes(buf);

    Ok(AuroraModel {
        input_layer: BitDense {
            weights: input_weights,
            biases: input_biases,
        },
        output_weights,
        output_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_index_range() {
        // All feature indices should be in [0, 768)
        for &color in &[Color::White, Color::Black] {
            for &piece in &Piece::ALL {
                for sq_idx in 0..64 {
                    let sq = Square::from_index(sq_idx);
                    let white_idx = AuroraState::feature_index_white(piece, color, sq);
                    let black_idx = AuroraState::feature_index_black(piece, color, sq);
                    assert!(white_idx < INPUTS, "White feature index {} out of range", white_idx);
                    assert!(black_idx < INPUTS, "Black feature index {} out of range", black_idx);
                }
            }
        }
    }

    #[test]
    fn test_feature_indices_symmetric() {
        // For the starting position, white and black should have mirrored feature sets
        // A white pawn on e2 from white's view = black pawn on e7 from black's view
        let white_pawn_e2 = AuroraState::feature_index_white(
            Piece::Pawn, Color::White, Square::E2
        );
        let black_pawn_e7_from_black = AuroraState::feature_index_black(
            Piece::Pawn, Color::Black, Square::E7
        );
        // These should be the same index (both are "our pawn on e2" from each perspective)
        assert_eq!(white_pawn_e2, black_pawn_e7_from_black);
    }
}
