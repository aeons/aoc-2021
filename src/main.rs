#![allow(unused)]
#![warn(unused_imports)]

use std::fmt::Display;
use std::fs::read_to_string;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use itertools::Itertools;
use nom::{character, IResult, Parser, ToUsize};
use nom::bytes::complete::tag;
use nom::multi::separated_list1;

fn main() -> Result<()> {
    day10::part2()
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

fn parse_usize(input: &str) -> IResult<&str, usize> {
    character::complete::u32.map(|u| u.to_usize()).parse(input)
}

fn parse_comma_separated_numbers(input: &str) -> IResult<&str, Vec<usize>> {
    separated_list1(tag(","), parse_usize)(input)
}

fn load_comma_separated_numbers(day: usize) -> Result<Vec<usize>> {
    let input = load_raw(day)?;
    let (_, numbers) = parse_comma_separated_numbers(&input).map_err(|e| e.to_owned())?;

    Ok(numbers)
}

fn print_answer<T: Display>(day: usize, part: usize, answer: T) -> Result<()> {
    println!("Day {:2}, part {}: {}", day, part, answer);
    Ok(())
}

mod day1 {
    use crate::*;

    fn calc(part: usize, data: Vec<i32>) -> usize {
        data.windows(2).filter(|s| s[1] > s[0]).count()
    }

    pub fn part1() -> Result<()> {
        let answer = calc(1, load(1)?);

        print_answer(1, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let data: Vec<i32> = load(1)?.windows(3).map(|w| w.iter().sum()).collect();
        let answer = calc(2, data);
        print_answer(1, 2, answer)
    }
}

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

mod day4 {
    use std::collections::HashSet;

    use nom::character::complete as c;
    use nom::IResult;
    use nom::multi::{many1, many_m_n, separated_list1};
    use nom::sequence::{preceded, terminated};

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
        pub fn new(numbers: Vec<usize>) -> Self {
            let mut rows: Vec<Row> = vec![Row::new(); 10];
            for (idx, number) in numbers.iter().enumerate() {
                rows[idx / 5].numbers.insert(*number);
                rows[idx % 5 + 5].numbers.insert(*number);
            }
            Board {
                numbers: HashSet::from_iter(numbers),
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

    fn parse_board(input: &str) -> IResult<&str, Board> {
        let (input, _) = c::line_ending(input)?;
        let row = preceded(c::space0, separated_list1(c::space1, parse_usize));
        let (input, rows) = many_m_n(5, 5, terminated(row, c::line_ending))(input)?;
        let numbers = rows.into_iter().flatten().collect();

        Ok((input, Board::new(numbers)))
    }

    fn parse_boards(input: &str) -> IResult<&str, Vec<Board>> {
        let (input, boards) = many1(parse_board)(input)?;

        Ok((input, boards))
    }

    fn load_data() -> Result<(Vec<usize>, Vec<Board>)> {
        let input = load_raw(4)?;
        let (input, numbers) = terminated(parse_comma_separated_numbers, c::line_ending)(&input)
            .map_err(|e| e.to_owned())?;
        let (_, boards) = parse_boards(input).map_err(|e| e.to_owned())?;

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
        let (numbers, mut boards) = load_data()?;

        let mut called_numbers = HashSet::new();
        let mut winner = None;

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
            .ok_or(anyhow!("no winner found"))?;

        print_answer(4, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let (numbers, mut boards) = load_data()?;

        let mut called_numbers = HashSet::new();
        let mut last_winner = None;

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
            .ok_or(anyhow!("no winner found"))?;

        print_answer(4, 2, answer)
    }
}

mod day5 {
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};

    use nom::bytes::streaming::tag;
    use nom::character::complete as c;
    use nom::IResult;
    use nom::multi::separated_list1;
    use nom::sequence::separated_pair;

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

mod day6 {
    use nom::ToUsize;

    use crate::*;

    fn load_data() -> Result<Vec<usize>> {
        let input = load_raw(6)?;
        let (_, lantern_fish) =
            parse_comma_separated_numbers(input.as_ref()).map_err(|e| e.to_owned())?;

        Ok(lantern_fish)
    }

    fn breed_lantern_fish(days: usize) -> Result<usize> {
        let mut fish = [0; 9];
        let lantern_fish: Vec<usize> = load_data()?.iter().map(|f| f.to_usize()).collect();
        for lf in lantern_fish {
            fish[lf] += 1;
        }

        for _ in 0..days {
            let zero_fish = fish[0];
            for i in 0..8 {
                fish[i] = fish[i + 1];
            }
            fish[6] += zero_fish;
            fish[8] = zero_fish;
        }

        Ok(fish.iter().sum())
    }

    pub fn part1() -> Result<()> {
        print_answer(6, 1, breed_lantern_fish(80)?)
    }

    pub fn part2() -> Result<()> {
        print_answer(6, 2, breed_lantern_fish(256)?)
    }
}

mod day7 {
    use crate::*;

    fn move_crabs<F: Fn(usize) -> usize>(calc_fuel: F) -> Result<usize> {
        let input = load_comma_separated_numbers(7)?;

        let mut crabs = vec![0; *input.iter().max().unwrap_or(&0) + 1];

        for crab in input {
            crabs[crab] += 1;
        }

        let mut min_fuel = usize::MAX;

        'target_pos_loop: for target_pos in 0..crabs.len() {
            let mut fuel = 0;
            for crab_pos in 0..crabs.len() {
                let diff = (crab_pos as i64 - target_pos as i64).abs();
                fuel += calc_fuel(diff as usize) * crabs[crab_pos];
                if fuel > min_fuel {
                    continue 'target_pos_loop;
                }
            }

            if fuel < min_fuel {
                min_fuel = fuel;
            }
        }

        Ok(min_fuel)
    }

    pub fn part1() -> Result<()> {
        let answer = move_crabs(|diff| diff)?;
        print_answer(6, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let answer = move_crabs(|diff| diff * (diff + 1) / 2)?;
        print_answer(6, 2, answer)
    }
}

mod day8 {
    use std::collections::HashSet;

    use anyhow::anyhow;
    use nom::bytes::complete::take_while_m_n;
    use nom::character::complete::line_ending;
    use nom::character::streaming::space1;

    use crate::*;

    type Segment = HashSet<char>;

    fn parse_segment(input: &str) -> IResult<&str, Segment> {
        take_while_m_n(1, 7, |c| c >= 'a' && c <= 'g')
            .map(|w: &str| HashSet::from_iter(w.chars()))
            .parse(input)
    }

    fn parse_line(input: &str) -> IResult<&str, (Vec<Segment>, Vec<Segment>)> {
        let (input, patterns) = separated_list1(space1, parse_segment)(input)?;
        let (input, _) = tag(" | ")(input)?;
        let (input, segments) = separated_list1(space1, parse_segment)(input)?;

        Ok((input, (patterns, segments)))
    }

    fn load_data() -> Result<Vec<(Vec<Segment>, Vec<Segment>)>> {
        let input = load_raw(8)?;
        let (input, data) =
            separated_list1(line_ending, parse_line)(&input).map_err(|e| e.to_owned())?;
        Ok(data)
    }

    fn find_and_pop<I, P: FnMut(&I) -> bool>(v: &mut Vec<I>, pred: P) -> Result<I> {
        if let Some(pos) = v.iter().position(pred) {
            Ok(v.swap_remove(pos))
        } else {
            Err(anyhow!("element not found"))
        }
    }

    fn deduce_numbers(patterns: &[Segment]) -> Result<Vec<HashSet<char>>> {
        let diff = |a: &HashSet<char>, b: &HashSet<char>| -> Result<char> {
            let c = a.difference(&b).next().ok_or(anyhow!("diff is empty"))?;
            Ok(*c)
        };

        let mut patterns = patterns.to_vec();

        let one = find_and_pop(&mut patterns, |p| p.len() == 2)?;
        let four = find_and_pop(&mut patterns, |p| p.len() == 4)?;
        let seven = find_and_pop(&mut patterns, |p| p.len() == 3)?;
        let eight = find_and_pop(&mut patterns, |p| p.len() == 7)?;

        let two = find_and_pop(&mut patterns, |p| {
            p.len() == 5 && p.difference(&four).count() == 3
        })?;
        let three = find_and_pop(&mut patterns, |p| {
            p.len() == 5 && p.difference(&one).count() == 3
        })?;
        let five = find_and_pop(&mut patterns, |p| p.len() == 5)?;

        let nine = find_and_pop(&mut patterns, |p| {
            p.len() == 6 && p.difference(&three).count() == 1
        })?;
        let zero = find_and_pop(&mut patterns, |p| {
            p.len() == 6 && p.difference(&seven).count() == 3
        })?;
        let six = find_and_pop(&mut patterns, |p| p.len() == 6)?;

        Ok([zero, one, two, three, four, five, six, seven, eight, nine].into())
    }

    pub fn part1() -> Result<()> {
        let data = load_data()?;

        let lengths: HashSet<usize> = HashSet::from([2, 3, 4, 7]);

        let answer: usize = data
            .into_iter()
            .map(|(_, segments)| {
                segments
                    .iter()
                    .filter(|s| lengths.contains(&s.len()))
                    .count()
            })
            .sum();

        print_answer(8, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let data = load_data()?;

        let to_number = |ss: &Vec<Segment>, mapping: &Vec<HashSet<char>>| -> Result<usize> {
            let to_digit = |s: &Segment| {
                mapping
                    .iter()
                    .position(|d| d == s)
                    .ok_or(anyhow!("digit not found"))
            };
            let decoded = ss.iter().map(to_digit).collect::<Result<Vec<_>>>()?;
            Ok(decoded[0] * 1000 + decoded[1] * 100 + decoded[2] * 10 + decoded[3])
        };

        let numbers = data
            .iter()
            .map(|(patterns, segments)| {
                let mapping = deduce_numbers(patterns)?;
                to_number(segments, &mapping)
            })
            .collect::<Result<Vec<_>>>()?;

        let answer: usize = numbers.iter().sum();

        print_answer(8, 2, answer)
    }
}

mod day9 {
    use std::collections::HashSet;

    use nom::bytes::complete::take_while_m_n;
    use nom::character::{complete as c};
    use nom::combinator::map_parser;
    use nom::multi::many1;

    use crate::*;

    #[derive(Debug)]
    struct HeightMap {
        heights: Vec<usize>,
        width: usize,
        height: usize,
    }

    impl HeightMap {
        fn parse_line(input: &str) -> IResult<&str, Vec<usize>> {
            many1(map_parser(
                take_while_m_n(1, 1, |c: char| c.is_ascii_digit()),
                c::u64.map(|d| d.to_usize()),
            ))(input)
        }

        fn parse_heights(input: &str) -> IResult<&str, Vec<Vec<usize>>> {
            separated_list1(c::line_ending, Self::parse_line)(input)
        }

        pub fn load() -> Result<HeightMap> {
            let input = load_raw(9)?;
            let (_, heights) = Self::parse_heights(&input).map_err(|e| e.to_owned())?;
            let width = heights[0].len();
            let height = heights.len();
            let heights = heights.into_iter().flatten().collect();

            Ok(HeightMap {
                heights,
                width,
                height,
            })
        }

        pub fn get(&self, x: usize, y: usize) -> usize {
            self.heights[x + self.width * y]
        }

        pub fn neighbours(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
            let mut ns = Vec::new();
            if x > 0 {
                ns.push((x - 1, y))
            }
            if x < self.width - 1 {
                ns.push((x + 1, y))
            }
            if y > 0 {
                ns.push((x, y - 1))
            }
            if y < self.height - 1 {
                ns.push((x, y + 1))
            }
            ns
        }

        pub fn low_points(&self) -> Vec<(usize, usize)> {
            (0..self.width)
                .cartesian_product(0..self.height)
                .filter(|&(x, y)| {
                    let h = self.get(x, y);
                    self.neighbours(x, y)
                        .iter()
                        .map(|&(x, y)| self.get(x, y))
                        .all(|n| h < n)
                })
                .collect()
        }

        pub fn basin_size(&self, start_x: usize, start_y: usize) -> usize {
            let mut q = vec![(start_x, start_y)];
            let mut basin = HashSet::new();
            basin.insert((start_x, start_y));

            while let Some((x, y)) = q.pop() {
                let mut valid_neighbours = self
                    .neighbours(x, y)
                    .into_iter()
                    .filter(|&(nx, ny)| !basin.contains(&(nx, ny)) && self.get(nx, ny) < 9)
                    .collect_vec();
                basin.extend(valid_neighbours.iter());
                q.append(&mut valid_neighbours);
            }

            basin.len()
        }
    }

    pub fn part1() -> Result<()> {
        let height_map = HeightMap::load()?;

        let answer: usize = height_map
            .low_points()
            .iter()
            .map(|&(x, y)| height_map.get(x, y) + 1)
            .sum();

        print_answer(9, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let height_map = HeightMap::load()?;

        let answer: usize = height_map
            .low_points()
            .iter()
            .map(|&(x, y)| height_map.basin_size(x, y))
            .sorted()
            .rev()
            .take(3)
            .product();

        print_answer(9, 2, answer)
    }
}

mod day10 {
    use crate::*;

    #[derive(Debug)]
    enum SyntaxCheck {
        Ok,
        Incomplete(Vec<char>),
        Corrupted(char),
        Err(anyhow::Error),
    }

    fn check_line(line: &str) -> SyntaxCheck {
        let opening = |c: char| match c {
            ')' => Some('('),
            ']' => Some('['),
            '}' => Some('{'),
            '>' => Some('<'),
            _ => None
        };

        let mut stack = Vec::new();


        for c in line.chars() {
            match c {
                '(' | '[' | '{' | '<' => stack.push(c),
                ')' | ']' | '}' | '>' => {
                    let open = stack.pop();
                    if open != opening(c) {
                        return SyntaxCheck::Corrupted(c);
                    }
                }
                _ => return SyntaxCheck::Err(anyhow!("unknown character '{}'", c))
            }
        }

        if stack.is_empty() {
            SyntaxCheck::Ok
        } else {
            SyntaxCheck::Incomplete(stack)
        }
    }

    pub fn part1() -> Result<()> {
        let lines: Vec<String> = load(10)?;

        let score = |c| match c {
            ')' => 3,
            ']' => 57,
            '}' => 1197,
            '>' => 25137,
            _ => 0
        };

        let answer: usize = lines.iter()
            .map(|line| check_line(line))
            .filter_map(|check| match check {
                SyntaxCheck::Corrupted(c) => Some(score(c)),
                _ => None
            })
            .sum();

        print_answer(10, 1, answer)
    }

    pub fn part2() -> Result<()> {
        let lines: Vec<String> = load(10)?;

        let score = |chars: Vec<char>| chars.iter().rev()
            .map(|c| match c {
                '(' => 1,
                '[' => 2,
                '{' => 3,
                '<' => 4,
                _ => 0
            }).fold(0, |acc, score| acc * 5 + score);

        let scores: Vec<usize> = lines.iter()
            .map(|line| check_line(line))
            .filter_map(|check| match check {
                SyntaxCheck::Incomplete(chars) => Some(score(chars)),
                _ => None
            }).sorted().collect();

        print_answer(10, 2, scores[scores.len() / 2])
    }
}