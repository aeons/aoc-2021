#![allow(unused)]

use std::fmt::Display;
use std::fs::read_to_string;
use std::str::FromStr;

use anyhow::Result;

fn main() -> Result<()> {
    day5::part1()
}

fn load_raw(day: usize) -> Result<String> {
    let data = read_to_string(format!("data/{:02}.txt", day))?;
    Ok(data)
}

fn load<T>(day: usize) -> Result<Vec<T>>
where
    T: FromStr,
    <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let raw_data = read_to_string(format!("data/{:02}.txt", day))?;
    let lines = raw_data.trim().lines();
    lines
        .map(|line| line.parse().map_err(anyhow::Error::from))
        .collect()
}

#[derive(thiserror::Error, Debug)]
#[error("failed to parse input")]
pub struct ParseError {
    source: anyhow::Error,
}

impl ParseError {
    pub fn msg(s: &'static str) -> ParseError {
        ParseError {
            source: anyhow::Error::msg(s),
        }
    }
}

fn print_answer<T: Display>(day: usize, part: usize, answer: T) -> Result<()> {
    Ok(println!("Day {:2}, part {}: {}", day, part, answer))
}

//<editor-fold desc="Day 1">
mod day1 {
    use crate::*;

    fn calc(part: usize, data: Vec<i32>) -> usize {
        data.iter()
            .zip(data.iter().skip(1))
            .filter(|(a, b)| b > a)
            .count()
    }

    pub fn part1() -> Result<()> {
        let answer = calc(1, load(1)?);

        print_answer(1, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let data: Vec<i32> = load(1)?
            .as_slice()
            .windows(3)
            .map(|w| w.iter().sum())
            .collect();
        let answer = calc(2, data);
        print_answer(1, 2, answer)
    }
}
//</editor-fold>

//<editor-fold desc="Day 2">
mod day2 {
    use std::fmt::Debug;
    use std::str::FromStr;

    use crate::*;

    #[derive(Debug)]
    enum Direction {
        Forward(usize),
        Down(usize),
        Up(usize),
    }

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    pub struct DirectionParseError(String);

    impl FromStr for Direction {
        type Err = DirectionParseError;

        fn from_str(s: &str) -> Result<Direction, Self::Err> {
            let parse = |n: &str, f: fn(usize) -> Direction| {
                n.parse::<usize>()
                    .map(f)
                    .map_err(|err| DirectionParseError(err.to_string()))
            };

            let parts: Vec<&str> = s.split_whitespace().collect();
            match parts.as_slice() {
                ["forward", n] => parse(n, Direction::Forward),
                ["down", n] => parse(n, Direction::Down),
                ["up", n] => parse(n, Direction::Up),
                _ => Err(DirectionParseError("failed to parse direction".to_owned())),
            }
        }
    }

    pub fn part1() -> Result<()> {
        let mut horiz = 0;
        let mut depth = 0;

        for direction in load(2)? {
            match direction {
                Direction::Forward(n) => horiz += n,
                Direction::Down(n) => depth += n,
                Direction::Up(n) => depth -= n,
            }
        }

        print_answer(2, 1, horiz * depth)
    }

    pub fn part2() -> Result<()> {
        let mut horiz = 0;
        let mut depth = 0;
        let mut aim = 0;

        for direction in load::<Direction>(2)? {
            match direction {
                Direction::Forward(n) => {
                    horiz += n;
                    depth += aim * n;
                }
                Direction::Down(n) => aim += n,
                Direction::Up(n) => aim -= n,
            }
        }

        print_answer(2, 2, horiz * depth)
    }
}
//</editor-fold>

//<editor-fold desc="Day 3">
mod day3 {
    use crate::*;

    fn load() -> Result<Vec<usize>> {
        super::load::<String>(3)?
            .iter()
            .map(|s| usize::from_str_radix(s, 2).map_err(anyhow::Error::from))
            .collect()
    }

    fn is_bit_set(n: &usize, bit: usize) -> bool {
        *n & (1 << bit) != 0
    }

    fn filter_bits_set(numbers: Vec<usize>, bit: usize, is_set: bool) -> Vec<usize> {
        numbers
            .into_iter()
            .filter(move |n| is_bit_set(n, bit) == is_set)
            .collect()
    }

    pub fn part1() -> Result<()> {
        let data = load()?;
        let data_len = data.len();

        let mut bit_count = [0; 12];

        for mut number in data {
            for i in (0..12).rev() {
                bit_count[i] += number & 1;
                number >>= 1;
            }
        }

        let mut gamma_rate: usize = 0;
        for bit in bit_count {
            gamma_rate <<= 1;
            if bit > data_len / 2 {
                gamma_rate |= 1;
            }
        }

        let epsilon_rate = !gamma_rate & 0b1111_1111_1111;

        print_answer(3, 1, gamma_rate * epsilon_rate)
    }

    fn filter(i: usize, numbers: Vec<usize>, choose_most_common: bool) -> usize {
        let filtered_set = filter_bits_set(numbers.clone(), i, true);
        let bits_set = filtered_set.len();
        let bits_unset = numbers.len() - bits_set;

        let should_bit_be_set = match (choose_most_common, bits_set, bits_unset) {
            (true, s, u) if s >= u => true,
            (true, _, _) => false,
            (false, s, u) if s >= u => false,
            (false, _, _) => true,
        };

        let filtered = if should_bit_be_set {
            filtered_set
        } else {
            filter_bits_set(numbers, i, false)
        };

        match filtered.as_slice() {
            [n] => *n,
            _ => filter(i - 1, filtered, choose_most_common),
        }
    }

    pub fn part2() -> Result<()> {
        let data = load()?;
        let data_len = data.len();

        let oxy_gen_rating = filter(11, data.clone(), true);
        let co2_scrub_rating = filter(11, data, false);

        print_answer(3, 2, oxy_gen_rating * co2_scrub_rating)
    }
}
//</editor-fold>

//<editor-fold desc="Day 4">
mod day4 {
    use std::collections::HashSet;

    use crate::*;

    #[derive(Clone, Debug)]
    struct Row {
        numbers: HashSet<usize>,
        called: usize,
    }

    impl Row {
        pub fn new() -> Self {
            Row {
                numbers: HashSet::new(),
                called: 0,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct Board {
        numbers: HashSet<usize>,
        rows: Vec<Row>,
    }

    impl Board {
        pub fn new(numbers: &[usize]) -> Self {
            let mut rows: Vec<Row> = vec![Row::new(); 10];
            for (idx, number) in numbers.iter().enumerate() {
                rows[idx / 5].numbers.insert(*number);
                rows[idx % 5 + 5].numbers.insert(*number);
            }
            Board {
                numbers: HashSet::from_iter(numbers.to_owned()),
                rows,
            }
        }

        // Only returns true if this is the first row to win for this board
        pub fn call_number(&mut self, number: usize) -> bool {
            if self.has_won() {
                return false;
            }
            for mut row in &mut self.rows {
                if row.numbers.contains(&number) {
                    row.called += 1;
                }
                if row.called == 5 {
                    return true;
                }
            }
            return false;
        }

        fn has_won(&self) -> bool {
            self.rows.iter().find(|r| r.called == 5).is_some()
        }
    }

    fn parse_numbers<'a>(line: impl Iterator<Item = &'a str>) -> Result<Vec<usize>> {
        line.map(|n| n.parse::<usize>().map_err(anyhow::Error::from))
            .collect()
    }

    fn parse_boards(data: &[String]) -> Result<Vec<Board>> {
        let parse_lines = |lines: &[String]| -> Result<Board> {
            let one_line = lines.join(" ").trim().to_owned();
            let split_line = one_line.split_whitespace();
            let numbers = parse_numbers(split_line)?;
            Ok(Board::new(numbers.as_slice()))
        };
        data.chunks(6).map(parse_lines).collect()
    }

    fn parse_data() -> Result<(Vec<usize>, Vec<Board>)> {
        let data: Vec<String> = load(4)?;

        let first_line = data.first().ok_or(anyhow::Error::msg("no data"))?;
        let numbers = parse_numbers(first_line.split(','))?;
        let boards = parse_boards(&data[1..])?;

        Ok((numbers, boards))
    }

    fn calculate_score(
        board: &Board,
        called_numbers: &HashSet<usize>,
        winning_number: usize,
    ) -> usize {
        let sum_uncalled: usize = board.numbers.difference(called_numbers).sum();
        sum_uncalled * winning_number
    }

    pub fn part1() -> Result<()> {
        let (numbers, mut boards) = parse_data()?;

        let mut called_numbers = HashSet::new();
        let mut winner: Option<(Board, usize)> = None;

        'calling: for number in numbers {
            called_numbers.insert(number);
            for board in &mut boards {
                if board.call_number(number) {
                    winner = Some((board.clone(), number));
                    break 'calling;
                }
            }
        }

        let answer = winner
            .map(|(winning_board, last_called_number)| {
                calculate_score(&winning_board, &called_numbers, last_called_number)
            })
            .ok_or(anyhow::Error::msg("no winner found"))?;

        print_answer(4, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let (numbers, mut boards) = parse_data()?;

        let mut called_numbers = HashSet::new();
        let mut last_winner: Option<(Board, HashSet<usize>, usize)> = None;

        for number in numbers {
            called_numbers.insert(number);
            for board in &mut boards {
                if board.call_number(number) {
                    last_winner = Some((board.clone(), called_numbers.clone(), number));
                }
            }
        }

        let answer = last_winner
            .map(|(winning_board, called_numbers, last_called_number)| {
                calculate_score(&winning_board, &called_numbers, last_called_number)
            })
            .ok_or(anyhow::Error::msg("no winner found"))?;

        print_answer(4, 2, answer)
    }
}
//</editor-fold>

//<editor-fold desc="Day 5">
mod day5 {
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};

    use nom::bytes::streaming::tag;
    use nom::character::complete as c;
    use nom::multi::separated_list1;
    use nom::sequence::separated_pair;
    use nom::IResult;

    use crate::*;

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[derive(Debug, Copy, Clone)]
    struct Line {
        from: Point,
        to: Point,
    }

    impl Line {
        pub fn covered_points(&self, diagonal: bool) -> HashSet<Point> {
            if self.from.x == self.to.x {
                let lo = min(self.from.y, self.to.y);
                let hi = max(self.from.y, self.to.y);
                HashSet::from_iter((lo..=hi).map(|y| Point { x: self.from.x, y }))
            } else if self.from.y == self.to.y {
                let lo = min(self.from.x, self.to.x);
                let hi = max(self.from.x, self.to.x);
                HashSet::from_iter((lo..=hi).map(|x| Point { x, y: self.from.y }))
            } else if diagonal {
                let (start, end) = if self.from.x < self.to.x {
                    (self.from, self.to)
                } else {
                    (self.to, self.from)
                };
                let slope = if start.y < end.y { 1 } else { -1 };
                HashSet::from_iter((0..=(end.x - start.x)).map(|i| Point {
                    x: start.x + i,
                    y: start.y + i * slope,
                }))
            } else {
                HashSet::new()
            }
        }
    }

    fn parse_point(input: &str) -> IResult<&str, Point> {
        let (input, (x, y)) = separated_pair(c::i32, tag(","), c::i32)(input)?;
        Ok((input, Point { x, y }))
    }

    fn parse_line(input: &str) -> IResult<&str, Line> {
        let (input, (from, to)) = separated_pair(parse_point, tag(" -> "), parse_point)(input)?;
        Ok((input, Line { from, to }))
    }

    fn load_data() -> Result<Vec<Line>> {
        let data = load_raw(5)?;
        let (_, lines) =
            separated_list1(c::line_ending, parse_line)(&data).map_err(|e| e.to_owned())?; // anyhow::Error::from)?;
        Ok(lines)
    }

    fn covered_points(lines: Vec<Line>, diagonal: bool) -> usize {
        let covered_points = lines.iter().map(|l| l.covered_points(true)).fold(
            HashMap::new(),
            |mut covered, points| {
                for point in points {
                    // println!("inserting {:?}", point);
                    *covered.entry(point).or_insert(0) += 1
                }
                covered
            },
        );

        covered_points.into_iter().filter(|(_, n)| *n >= 2).count()
    }

    pub fn part1() -> Result<()> {
        let answer = covered_points(load_data()?, false);
        print_answer(5, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let answer = covered_points(load_data()?, true);
        print_answer(5, 2, answer)
    }
}
//</editor-fold>
