#![allow(unused)]

use std::fmt::Display;
use std::fs::read_to_string;
use anyhow::Result;

fn main() {
    day2::part2();
}

fn load_raw<T, F>(day: usize, f: F) -> Vec<T>
    where F: FnMut(&str) -> T {
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
    load_raw(day, |s| s.parse::<i32>().expect(format!("Failed to parse '{}' as an integer", s).as_str()))
}

fn print_answer<T: Display>(day: usize, part: usize, answer: T) {
    println!("Day {:2}, part {}: {}", day, part, answer)
}

// Day 1
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
        let data: Vec<i32> = load_ints(1).as_slice().windows(3).map(|w| w.iter().sum()).collect();
        calc(2, data);
    }
}

// Day 2
mod day2 {
    use std::num::ParseIntError;
    use std::str::{FromStr, SplitWhitespace};
    use crate::*;

    #[derive(Debug)]
    enum Direction {
        Forward(usize),
        Down(usize),
        Up(usize),
    }

    impl FromStr for Direction {
        type Err = anyhow::Error;

        fn from_str(s: &str) -> Result<Direction> {
            let parse = |n: &str, f: fn(usize) -> Direction|
                n.parse::<usize>().map(f).map_err(anyhow::Error::from);

            let parts: Vec<&str> = s.split_whitespace().collect();
            match parts.as_slice() {
                ["forward", n] => parse(n, Direction::Forward),
                ["down", n] => parse(n, Direction::Down),
                ["up", n] => parse(n, Direction::Up),
                _ => Err(anyhow::Error::msg("Failed to parse direction"))
            }
        }
    }

    fn load_directions() -> Vec<Direction> {
        load_strings(2).iter().map(|l| l.parse::<Direction>().unwrap()).collect()
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
