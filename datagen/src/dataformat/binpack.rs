use std::fmt::Display;

use super::CompressedChessBoard;

use monty::{Board, Castling, ChessState, GameState, Move};

pub struct Binpack {
    pub startpos: CompressedChessBoard,
    result: u8,
    moves: Vec<(u16, i16)>,
}

impl Display for Binpack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "fen: {}", Board::from(self.startpos).as_fen())?;
        writeln!(f, "result: {}", self.result)?;
        for &(mov, score) in self.moves.iter() {
            writeln!(f, "{}: {}", Move::from(mov).to_uci(&Castling::default()), score)?;
        }

        Ok(())
    }
}

impl Binpack {
    pub fn new(pos: ChessState) -> Self {
        Self {
            startpos: pos.into(),
            result: 3,
            moves: Vec::new(),
        }
    }

    pub fn as_uci_position_command(&self) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;
        let pos = self.startpos;
        let mut board = Board::from(pos);
        let mut stack = Vec::new();
        let castling = Castling::from_raw(&board, pos.rook_files()).ok_or(std::fmt::Error)?;

        let mut s = String::new();
        write!(&mut s, "position fen {} moves", board.as_fen())?;
        for &(mov, _) in self.moves.iter() {
            let m = Move::from(mov);
            assert_eq!(board.game_state(&castling, &stack), GameState::Ongoing);
            stack.push(board.hash());
            board.make(m, &castling);
            write!(&mut s, " {}", m.to_uci(&castling))?;
        }

        assert_ne!(board.game_state(&castling, &stack), GameState::Ongoing);

        Ok(s)
    }

    pub fn set_result(&mut self, result: f32) {
        self.result = (2.0 * result) as u8;
    }

    pub fn result(&self) -> u8 {
        self.result
    }

    pub fn push(&mut self, stm: usize, best_move: Move, mut score: f32) {
        if stm == 1 {
            score = 1.0 - score;
        }

        let score = -(400.0 * (1.0 / score - 1.0).ln()) as i16;

        self.moves.push((best_move.into(), score));
    }

    pub fn serialise_into(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writer.write_all(&self.startpos.as_bytes())?;
        writer.write_all(&[self.result])?;

        for (mov, score) in &self.moves {
            writer.write_all(&mov.to_le_bytes())?;
            writer.write_all(&score.to_le_bytes())?;
        }

        writer.write_all(&[0; 4])?;
        Ok(())
    }

    pub fn deserialise_from(
        reader: &mut impl std::io::BufRead,
        buffer: Vec<(u16, i16)>,
    ) -> std::io::Result<Self> {
        let mut startpos = [0; std::mem::size_of::<CompressedChessBoard>()];

        if let Ok(buf) = reader.fill_buf() {
            if buf.is_empty() {
                return Err(std::io::Error::new(std::io::ErrorKind::Other, "eof"));
            }
        } else {
            panic!("what");
        }

        reader.read_exact(&mut startpos).unwrap();
        let startpos = CompressedChessBoard::from_bytes(startpos);

        let mut result = [0];
        reader.read_exact(&mut result)?;
        let result = result[0];

        let mut moves = buffer;
        moves.clear();

        loop {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;

            if buf[0] == 0 && buf[1] == 0 {
                break;
            }

            let mov = u16::from_le_bytes([buf[0], buf[1]]);
            let score = i16::from_le_bytes([buf[2], buf[3]]);

            moves.push((mov, score));
        }

        Ok(Self {
            startpos,
            result,
            moves,
        })
    }

    pub fn deserialise_map<F>(reader: &mut impl std::io::BufRead, mut f: F) -> std::io::Result<()>
    where
        F: FnMut(&mut Board, &Castling, Move, i16, f32),
    {
        let mut startpos = [0; std::mem::size_of::<CompressedChessBoard>()];
        reader.read_exact(&mut startpos)?;
        let startpos = CompressedChessBoard::from_bytes(startpos);

        let mut result = [0];
        reader.read_exact(&mut result).unwrap();
        let result = f32::from(result[0]) / 2.0;

        let mut board = Board::from(startpos);
        let castling = Castling::from_raw(&board, startpos.rook_files());

        loop {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf).unwrap();

            if buf == [0; 4] {
                break;
            }

            let mov = u16::from_le_bytes([buf[0], buf[1]]);
            let score = i16::from_le_bytes([buf[2], buf[3]]);

            f(&mut board, &castling.unwrap(), mov.into(), score, result);
        }

        Ok(())
    }
}
