use std::io::{Error, ErrorKind, Write};

use crate::ataxx::{Board, Move};

pub struct SearchData {
    pub best_move: Move,
    pub score: f32,
    pub visit_distribution: Option<Vec<(Move, u32)>>,
}

impl SearchData {
    pub fn new(best_move: Move, score: f32, visit_distribution: Option<Vec<(Move, u32)>>) -> Self {
        let mut visit_distribution: Option<Vec<(Move, u32)>> =
            visit_distribution.map(|x| x.iter().map(|&(mov, visits)| (mov, visits)).collect());

        if let Some(dist) = visit_distribution.as_mut() {
            dist.sort_by_key(|(mov, _)| u16::from(*mov));
        }

        Self {
            best_move,
            score,
            visit_distribution,
        }
    }
}

macro_rules! read_primitive_into_vec {
    ($reader:expr, $writer:expr, $t:ty) => {{
        let mut buf = [0u8; std::mem::size_of::<$t>()];
        $reader.read_exact(&mut buf)?;
        $writer.extend_from_slice(&buf);
        <$t>::from_le_bytes(buf)
    }};
}

macro_rules! read_into_primitive {
    ($reader:expr, $t:ty) => {{
        let mut buf = [0u8; std::mem::size_of::<$t>()];
        $reader.read_exact(&mut buf)?;
        <$t>::from_le_bytes(buf)
    }};
}

pub struct MontyAtaxxFormat {
    pub moves: Vec<SearchData>,
    pub result: f32,
}

impl Default for MontyAtaxxFormat {
    fn default() -> Self {
        Self {
            moves: Vec::new(),
            result: 0.0,
        }
    }
}

impl MontyAtaxxFormat {
    pub fn push(&mut self, position_data: SearchData) {
        self.moves.push(position_data);
    }

    pub fn serialise_into_buffer(&self, writer: &mut Vec<u8>) -> std::io::Result<()> {
        if !writer.is_empty() {
            return Err(Error::new(ErrorKind::Other, "Buffer is not empty!"));
        }

        let result = (self.result * 2.0) as u8;
        writer.write_all(&result.to_le_bytes())?;

        for data in &self.moves {
            if data.score.clamp(0.0, 1.0) != data.score {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Score outside valid range!",
                ));
            }

            let score = (data.score * f32::from(u16::MAX)) as u16;

            writer.write_all(&u16::from(data.best_move).to_le_bytes())?;
            writer.write_all(&score.to_le_bytes())?;

            let num_moves = data
                .visit_distribution
                .as_ref()
                .map(|dist| dist.len())
                .unwrap_or(0) as u8;

            writer.write_all(&num_moves.to_le_bytes())?;

            if let Some(dist) = data.visit_distribution.as_ref() {
                let max_visits = dist
                    .iter()
                    .max_by_key(|(_, visits)| visits)
                    .map(|x| x.1)
                    .unwrap_or(0);
                for (_, visits) in dist {
                    let scaled_visits = if max_visits <= 256 {
                        *visits as u8
                    } else {
                        (*visits as f32 * 256.0 / max_visits as f32) as u8
                    };
                    writer.write_all(&scaled_visits.to_le_bytes())?;
                }
            }
        }

        writer.write_all(&[0; 2])?;
        Ok(())
    }

    pub fn deserialise_from(reader: &mut impl std::io::BufRead) -> std::io::Result<Self> {
        let result = read_into_primitive!(reader, u8) as f32 / 2.0;

        let mut moves = Vec::new();
        let mut pos = Board::default();

        loop {
            let best_move = Move::from(read_into_primitive!(reader, u16));

            if best_move == Move::NULL {
                break;
            }

            let score = f32::from(read_into_primitive!(reader, u16)) / f32::from(u16::MAX);

            let num_moves = read_into_primitive!(reader, u8);

            let visit_distribution = if num_moves == 0 {
                None
            } else {
                let mut dist = Vec::with_capacity(usize::from(num_moves));

                pos.map_legal_moves(|mov| dist.push((mov, 0)));
                dist.sort_by_key(|(mov, _)| u16::from(*mov));

                for entry in &mut dist {
                    entry.1 = u32::from(read_into_primitive!(reader, u8));
                }

                Some(dist)
            };

            moves.push(SearchData {
                best_move,
                score,
                visit_distribution,
            });

            pos.make(best_move);
        }

        Ok(Self { result, moves })
    }

    pub fn deserialise_fast_into_buffer(
        reader: &mut impl std::io::BufRead,
        buffer: &mut Vec<u8>,
    ) -> std::io::Result<()> {
        buffer.clear();

        let _ = read_primitive_into_vec!(reader, buffer, u8);

        loop {
            let best_move = Move::from(read_primitive_into_vec!(reader, buffer, u16));

            if best_move == Move::NULL {
                break;
            }

            let _ = read_primitive_into_vec!(reader, buffer, u16);

            let num_moves = read_primitive_into_vec!(reader, buffer, u8);

            for _ in 0..num_moves {
                let _ = read_primitive_into_vec!(reader, buffer, u8);
            }
        }

        Ok(())
    }
}
