use std::fmt;
use std::error::Error;
use std::path::Path;
use yaml_rust::{YamlLoader, yaml::Yaml};
use meval::Context;

use crate::constants::*;

pub enum InputError {
    InvalidInputFile(&'static str),
    CouldNotParse(String, String),
    //MissingField(String),
    MissingField(String, String),
}

impl fmt::Debug for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use InputError::*;
        let help_msg = "Usage: mpirun -n np ./opal input-file";
        match self {
            InvalidInputFile(s) => write!(f, "invalid input file: {}\n{}", s, help_msg),
            CouldNotParse(token,field) => write!(f, "unable to parse '{}' = '{}' in configuration file", token, field),
            MissingField(section,field) => write!(f, "unable to find '{}' in section '{}' with correct type in configuration file", field, section),
        }
    }
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for InputError {}

pub struct Configuration<'a> {
    input: Yaml,
    ctx: Context<'a>,
}

impl<'a> Configuration<'a> {
    pub fn from_file(path: &Path) -> Result<Configuration,InputError> {
        let contents = std::fs::read_to_string(path).map_err(|_e| InputError::InvalidInputFile("unable to read file"))?;
        let input = YamlLoader::load_from_str(&contents).map_err(|_e| InputError::InvalidInputFile("yaml trouble"))?;
        let input = input.first().ok_or(InputError::InvalidInputFile("yaml trouble"))?;
        Ok(Configuration {
            input: input.clone(),
            ctx: Context::new(),
        })
    }

    pub fn with_context(&mut self, section: &str) -> &mut Self {
        // Default constants and plasma-related functions
        self.ctx
            .var("m", ELECTRON_MASS)
            .var("me", ELECTRON_MASS)
            .var("mp", PROTON_MASS)
            .var("c", SPEED_OF_LIGHT)
            .var("e", ELEMENTARY_CHARGE)
            .var("eV", ELEMENTARY_CHARGE)
            .var("keV", 1.0e3 * ELEMENTARY_CHARGE)
            .var("MeV", 1.0e6 * ELEMENTARY_CHARGE)
            .var("femto", 1.0e-15)
            .var("pico", 1.0e-12)
            .var("nano", 1.0e-9)
            .var("micro", 1.0e-6)
            .var("milli", 1.0e-3)
            .func3("step", |x, min, max| if x >= min && x < max {1.0} else {0.0})
            .func("critical", |omega| VACUUM_PERMITTIVITY * ELECTRON_MASS * omega.powi(2) / ELEMENTARY_CHARGE.powi(2));

        // Read in from 'constants' block
        let tmp = self.ctx.clone(); // a constant cannot depend on other constants yet...
        //println!("{:#?}", self.input[section].as_hash());

        for (a, b) in self.input[section].as_hash().unwrap() {
            //println!("{:?} {:?}", a, b);
            match (a, b) {
                (Yaml::String(s), Yaml::Real(v)) => {
                    if let Ok(num) = v.parse::<f64>() {self.ctx.var(s, num);}
                },
                (Yaml::String(s), Yaml::String(v)) => {
                    if let Ok(expr) = v.parse::<meval::Expr>() {
                        if let Ok(num) = expr.eval_with_context(&tmp) {self.ctx.var(s, num);}
                    }
                },
                _ => ()
            }
        }

        self
    }

    pub fn real(&self, section: &str, field: &str) -> Result<f64, InputError> {
        let name = field.to_owned();
        match &self.input[section][field] {
            Yaml::Real(s) => s.parse::<f64>().map_err(|_| InputError::CouldNotParse(name.clone(), s.clone())),
            //Yaml::String(s) => s.parse::<meval::Expr>()?.eval_with_context(default_ctx),
            Yaml::String(s) => {
                let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(name.clone(), s.clone()))?; // Result<f64,meval:;err>
                expr.eval_with_context(&self.ctx).map_err(|_| InputError::CouldNotParse(name.clone(), s.clone()))
            },
            _ => Err(InputError::MissingField(section.to_owned(), name)),
        }
    }

    pub fn func(&'a self, section: &str, field: &str, arg: &str) -> Result<impl Fn(f64) -> f64 + 'a, InputError> {
        //println!("{:#?}", &self.input[section][field]);
        match &self.input[section][field] {
            Yaml::String(s) | Yaml::Real(s) => {
                let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
                let func = expr.bind_with_context(&self.ctx, arg).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
                Ok(func)
            },
            _ => Err(InputError::MissingField(section.to_owned(), field.to_owned()))
        }
    }

    pub fn func2(&'a self, section: &str, field: &str, args: [&str; 2]) -> Result<impl Fn(f64, f64) -> f64 + 'a, InputError> {
        match &self.input[section][field] {
            Yaml::String(s) | Yaml::Real(s) => {
                let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
                expr.bind2_with_context(&self.ctx, args[0], args[1]).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
            },
            _ => Err(InputError::MissingField(section.to_owned(), field.to_owned()))
        }
    }

    pub fn func3(&'a self, section: &str, field: &str, args: [&str; 3]) -> Result<impl Fn(f64, f64, f64) -> f64 + 'a, InputError> {
        match &self.input[section][field] {
            Yaml::String(s) | Yaml::Real(s) => {
                let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
                expr.bind3_with_context(&self.ctx, args[0], args[1], args[2]).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
            },
            _ => Err(InputError::MissingField(section.to_owned(), field.to_owned()))
        }
    }

    pub fn integer(&self, section: &str, field: &str) -> Result<i64, InputError> {
        match &self.input[section][field] {
            Yaml::Integer(i) => Ok(*i),
            _ => Err(InputError::MissingField(section.to_owned(), field.to_owned())),
        }
    }
    
    pub fn bool(&self, section: &str, field: &str) -> Result<bool, InputError> {
        match &self.input[section][field] {
            Yaml::Boolean(b) => Ok(*b),
            _ => Err(InputError::MissingField(section.to_owned(), field.to_owned())),
        }
    }
    
    pub fn strings(&self, section: &str, field: &str) -> Result<Vec<String>, InputError> {
        let name = field.to_owned();
        match &self.input[section][field] {
            Yaml::String(s) => {
                Ok(vec![s.clone()])
            },
            Yaml::Array(array) => {
                // a is a vec of Vec<Yaml>
                let take_yaml_string = |y: &Yaml| -> Option<String> {
                    match y {
                        Yaml::String(s) => Some(s.clone()),
                        _ => None
                    }
                };
                let got: Vec<String> = array.iter().filter_map(take_yaml_string).collect();
                if got.is_empty() {
                    Err(InputError::CouldNotParse(section.to_owned(), name))
                } else {
                    Ok(got)
                }
            },
            _ => Err(InputError::MissingField(section.to_owned(), name))
        }
    }
    
    pub fn string(&self, section: &str, field: &str) -> Result<String, InputError> {
        let strs = self.strings(section, field)?;
        //let str = strs.first().ok_or(InputError::MissingField(field.to_owned()))?;
        //str.clone()
        Ok(strs[0].clone())
    }
}

/*
pub fn read_evaluate<C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, ctx: &C) -> Result<f64,InputError> {
    let name = field.to_owned();
    match &input[section][field] {
        Yaml::Real(s) => s.parse::<f64>().map_err(|_| InputError::CouldNotParse(name.clone(), s.clone())),
        //Yaml::String(s) => s.parse::<meval::Expr>()?.eval_with_context(default_ctx),
        Yaml::String(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(name.clone(), s.clone()))?; // Result<f64,meval:;err>
            expr.eval_with_context(ctx).map_err(|_| InputError::CouldNotParse(name.clone(), s.clone()))
        },
        _ => Err(InputError::MissingField(name)),
    }  
}

pub fn read_func1<'a, C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, arg: &str, ctx: &'a C) -> Result<impl Fn(f64) -> f64 + 'a,InputError> {
    match &input[section][field] {
        Yaml::String(s) | Yaml::Real(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
            expr.bind_with_context(ctx, arg).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
        },
        _ => Err(InputError::MissingField(field.to_owned()))
    }
}

pub fn read_func2<'a, C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, args: [&str; 2], ctx: &'a C) -> Result<impl Fn(f64, f64) -> f64 + 'a,InputError> {
    match &input[section][field] {
        Yaml::String(s) | Yaml::Real(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
            expr.bind2_with_context(ctx, args[0], args[1]).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
        },
        _ => Err(InputError::MissingField(field.to_owned()))
    }
}

pub fn read_func3<'a, C: meval::ContextProvider>(input: &Yaml, section: &str, field: &str, args: [&str; 3], ctx: &'a C) -> Result<impl Fn(f64, f64, f64) -> f64 + 'a,InputError> {
    match &input[section][field] {
        Yaml::String(s) | Yaml::Real(s) => {
            let expr = s.parse::<meval::Expr>().map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))?;
            expr.bind3_with_context(ctx, args[0], args[1], args[2]).map_err(|_| InputError::CouldNotParse(field.to_owned(), s.clone()))
        },
        _ => Err(InputError::MissingField(field.to_owned()))
    }
}

pub fn read_integer(input: &Yaml, section: &str, field: &str) -> Result<i64,InputError> {
    match &input[section][field] {
        Yaml::Integer(i) => Ok(*i),
        _ => Err(InputError::MissingField(field.to_owned())),
    }
}

pub fn read_bool(input: &Yaml, section: &str, field: &str) -> Result<bool,InputError> {
    match &input[section][field] {
        Yaml::Boolean(b) => Ok(*b),
        _ => Err(InputError::MissingField(field.to_owned())),
    }
}

pub fn read_strings(input: &Yaml, section: &str, field: &str) -> Result<Vec<String>, InputError> {
    let name = field.to_owned();
    match &input[section][field] {
        Yaml::String(s) => {
            Ok(vec![s.clone()])
        },
        Yaml::Array(array) => {
            // a is a vec of Vec<Yaml>
            let take_yaml_string = |y: &Yaml| -> Option<String> {
                match y {
                    Yaml::String(s) => Some(s.clone()),
                    _ => None
                }
            };
            let got: Vec<String> = array.iter().filter_map(take_yaml_string).collect();
            if got.is_empty() {
                Err(InputError::CouldNotParse(section.to_owned(), name))
            } else {
                Ok(got)
            }
        },
        _ => Err(InputError::MissingField(name))
    }
}

pub fn read_string(input: &Yaml, section: &str, field: &str) -> Result<String, InputError> {
    let strs = read_strings(input, section, field)?;
    //let str = strs.first().ok_or(InputError::MissingField(field.to_owned()))?;
    //str.clone()
    Ok(strs[0].clone())
}

// return a result type for error handling
pub fn read_to_context(input: &Yaml, section: &str, ctx: &mut meval::Context) {
    let tmp = ctx.clone();
    //println!("{:?}", input["constants"].as_hash());
    for (a, b) in input[section].as_hash().unwrap() {
        //println!("{:?} {:?}", a, b);
        match (a, b) {
            (Yaml::String(s), Yaml::Real(v)) => {
                if let Ok(num) = v.parse::<f64>() {ctx.var(s, num);}
            },
            (Yaml::String(s), Yaml::String(v)) => {
                if let Ok(expr) = v.parse::<meval::Expr>() {
                    if let Ok(num) = expr.eval_with_context(&tmp) {ctx.var(s, num);}
                }
            },
            _ => ()
        }
    }
}
*/