#![allow(unused)]

use anyhow::Result;
use std::fmt::Display;
use std::fs::read_to_string;

fn main() {
    day3::part2();
}

fn load_raw<T, F>(day: usize, f: F) -> Vec<T>
where
    F: FnMut(&str) -> T,
{
    read_to_string(format!("data/{:02}.txt", day))
        .expect(format!("Could not find data file for day {}", day).as_str())
        .trim()
        .lines()
        .filter(|l| !l.is_empty())
        .map(f)
        .collect()
}

fn load_strings(day: usize) -> Vec<String> {
    load_raw(day, |s| s.to_owned())
}

fn load_ints(day: usize) -> Vec<i32> {
    load_raw(day, |s| {
        s.parse::<i32>()
            .expect(format!("Failed to parse '{}' as an integer", s).as_str())
    })
}

fn load_radix_2(day: usize) -> Vec<usize> {
    load_raw(day, |s| {
        usize::from_str_radix(s, 2)
            .expect(format!("Failed to parse '{}' as a binary number", s).as_str())
    })
}

fn print_answer<T: Display>(day: usize, part: usize, answer: T) {
    println!("Day {:2}, part {}: {}", day, part, answer)
}

mod day1 {
    use crate::*;

    fn calc(part: usize, data: Vec<i32>) {
        let answer = data
            .iter()
            .zip(data.iter().skip(1))
            .filter(|(a, b)| b > a)
            .count();

        print_answer(1, part, answer);
    }

    pub fn part1() {
        calc(1, load_ints(1));
    }

    pub fn part2() {
        let data: Vec<i32> = load_ints(1)
            .as_slice()
            .windows(3)
            .map(|w| w.iter().sum())
            .collect();
        calc(2, data);
    }
}

mod day2 {
    use crate::*;
    use std::num::ParseIntError;
    use std::str::{FromStr, SplitWhitespace};

    #[derive(Debug)]
    enum Direction {
        Forward(usize),
        Down(usize),
        Up(usize),
    }

    impl FromStr for Direction {
        type Err = anyhow::Error;

        fn from_str(s: &str) -> Result<Direction> {
            let parse = |n: &str, f: fn(usize) -> Direction| {
                n.parse::<usize>().map(f).map_err(anyhow::Error::from)
            };

            let parts: Vec<&str> = s.split_whitespace().collect();
            match parts.as_slice() {
                ["forward", n] => parse(n, Direction::Forward),
                ["down", n] => parse(n, Direction::Down),
                ["up", n] => parse(n, Direction::Up),
                _ => Err(anyhow::Error::msg("Failed to parse direction")),
            }
        }
    }

    fn load_directions() -> Vec<Direction> {
        load_strings(2)
            .iter()
            .map(|l| l.parse::<Direction>().unwrap())
            .collect()
    }

    pub fn part1() {
        let mut horiz = 0;
        let mut depth = 0;

        for direction in load_directions() {
            match direction {
                Direction::Forward(n) => horiz += n,
                Direction::Down(n) => depth += n,
                Direction::Up(n) => depth -= n,
            }
        }

        print_answer(2, 1, horiz * depth)
    }

    pub fn part2() {
        let mut horiz = 0;
        let mut depth = 0;
        let mut aim = 0;

        for direction in load_directions() {
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

    fn is_bit_set(n: &usize, bit: usize) -> bool {
        *n & (1 << bit) != 0
    }

    fn filter_bits_set(numbers: Vec<usize>, bit: usize, is_set: bool) -> Vec<usize> {
        numbers
            .into_iter()
            .filter(move |n| is_bit_set(n, bit) == is_set)
            .collect()
    }

    pub fn part1() {
        let data = load_radix_2(3);
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

        print_answer(3, 1, gamma_rate * epsilon_rate);
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

    pub fn part2() {
        let data = load_radix_2(3);
        let data_len = data.len();

        let oxy_gen_rating = filter(11, data.clone(), true);
        let co2_scrub_rating = filter(11, data, false);

        print_answer(3, 2, oxy_gen_rating * co2_scrub_rating);
    }
}
