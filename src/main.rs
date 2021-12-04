#![allow(unused)]

use anyhow::Result;
use std::fmt::Display;
use std::fs::read_to_string;
use std::str::FromStr;

fn main() {
    day3::part2();
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
    use crate::*;
    use std::fmt::{Debug, Formatter};
    use std::num::ParseIntError;
    use std::str::{FromStr, SplitWhitespace};

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
