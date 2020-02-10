use std::fmt;
use mpi::traits::*;
use mpi::collective::SystemOperation;
use fitsio::*;

#[derive(Copy,Clone,PartialEq)]
pub enum BinSpec {
    Automatic,
    LogScaled,
    FixedNumber(usize),
    FixedSize(f64),
}

impl fmt::Display for BinSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinSpec::Automatic => write!(f, "BinSpec:Automatic"),
            BinSpec::LogScaled => write!(f, "BinSpec:LogScaled"),
            BinSpec::FixedNumber(n) => write!(f, "BinSpec:FixedNumber({})", n),
            BinSpec::FixedSize(dx) => write!(f, "BinSpec:FixedSize({})", dx),
        }
    }
}

impl From<&str> for BinSpec {
    fn from(s: &str) -> Self {
        if let Ok(nbins) = s.parse::<usize>() {
            BinSpec::FixedNumber(nbins)
        } else if let Ok(dx) = s.parse::<f64>() {
            BinSpec::FixedSize(dx)
        } else if s == "auto" {
            BinSpec::Automatic
        } else if s == "log" {
            BinSpec::LogScaled
        } else {
            BinSpec::Automatic
        }
    }
}

#[derive(Copy,Clone,PartialEq)]
pub enum HeightSpec {
    Count,
    Density,
    ProbabilityDensity,
}

impl From<&str> for HeightSpec {
    fn from(s: &str) -> Self {
        match s {
            "count" => HeightSpec::Count,
            "density" | "auto" => HeightSpec::Density,
            "probablity_density" | "pdf" => HeightSpec::ProbabilityDensity,
            _ => HeightSpec::Density,
        }
    }
}

impl fmt::Display for HeightSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            HeightSpec::Count => write!(f, "count"),
            HeightSpec::Density => write!(f, "density"),
            HeightSpec::ProbabilityDensity => write!(f, "pdf"),
        }
    }

}

#[allow(unused)]
pub struct Histogram {
    dim: usize,
    total: f64,
    bin_vol: f64,
    min: Vec<f64>,
    max: Vec<f64>,
    cts: Vec<f64>,
    bins: Vec<usize>,
    bin_sz: Vec<f64>,
    name: String,
    bunit: String,
    axis: Vec<String>,
    unit: Vec<String>,
}

fn min_max_by<T>(base: &[T], f: &impl Fn(&T) -> f64, wrapper: impl Fn(f64) -> f64) -> Option<(f64, f64)> {
    if base.is_empty() {
        None
    } else {
        let mut min = wrapper(f(base.first().unwrap()));
        let mut max = min;
        for t in base.iter() {
            let v = wrapper(f(t));
            if v.is_finite() {
                /*
                if v < min {
                    min = v;
                } else if v > max {
                    max = v;
                }
                */
                min = min.min(v); // avoid explicit branching, saves time if nthreads >= 4
                max = max.max(v);
            }
        }
        Some((min, max))
    }
}

fn linear_bin_vol(min: f64, bin_sz: f64, bin: usize) -> f64 {
    (min + (bin as f64) * bin_sz).exp() * bin_sz.exp_m1()
}

fn number_of_bins(min: f64, max: f64, n: usize, bspec: BinSpec) -> usize {
    if min == max {
        1
    } else {
        match bspec {
            BinSpec::Automatic | BinSpec::LogScaled =>
                (2.0 * (n as f64).cbrt()).ceil() as usize,
            BinSpec::FixedNumber(n) =>
                n,
            BinSpec::FixedSize(dx) =>
                ((max - min) / dx).ceil() as usize,
        }
    }
}

fn bin_size_and_volume(dim: usize, min: &[f64], max: &[f64], nbins: &[usize], bspec: &[BinSpec]) -> (Vec<f64>, f64) {
    let mut size: Vec<f64> = Vec::new();
    let mut volume = 1.0;
    for i in 0..dim {
        if min[i] == max[i] {
            volume *= 1.0;
            size.push(0.0);
        } else {
            let dx = match bspec[i] {
                BinSpec::Automatic | BinSpec::LogScaled | BinSpec::FixedNumber(_) =>
                    (max[i] - min[i]) / (nbins[i] as f64),
                BinSpec::FixedSize(dx) =>
                    dx,
            };
            volume *= dx;
            size.push(dx);
        }
    }
    (size, volume)
}

impl Histogram {
    pub fn generate_1d<T>(
        comm: &impl Communicator,
        base: &[T], accessor: &impl Fn(&T) -> f64, weight: &impl Fn(&T) -> f64,
        name: &str, unit: &str,
        bspec: BinSpec, hspec: HeightSpec) -> Option<Histogram> {
        //let rank = comm.rank();

        // Local min and max
        // Adjust for log-scaling!
        let (min, max) = if bspec == BinSpec::LogScaled {
            min_max_by(base, accessor, f64::ln).unwrap_or((std::f64::MAX, -std::f64::MAX))
        } else {
            min_max_by(base, accessor, std::convert::identity).unwrap_or((std::f64::MAX, -std::f64::MAX))
        };
        //let min = base.iter().map(accessor).min_by(|a,b| a.partial_cmp(b).unwrap() ).unwrap_or(std::f64::MAX);
        //let max = base.iter().map(accessor).max_by(|a,b| a.partial_cmp(b).unwrap() ).unwrap_or(-std::f64::MAX);
        //println!("{}: Local min = {:e}, max = {:e}, num = {}", rank, min, max, base.len());

        //Global min and max
        let mut gmin = 0.0;
        let mut gmax = 0.0;
        let mut gnum: usize = 0;
        comm.all_reduce_into(&min, &mut gmin, SystemOperation::min());
        comm.all_reduce_into(&max, &mut gmax, SystemOperation::max());
        comm.all_reduce_into(&base.len(), &mut gnum, SystemOperation::sum());
        //println!("{}: Global min = {:e} and max = {:e}, num = {}", rank, gmin, gmax, gnum);

        if gnum == 0 {
            return None;
        }
        
        // Prep bins
        let nbins = number_of_bins(gmin, gmax, gnum, bspec);

        let bin_vol = if gmin == gmax {
            1.0
        } else {
            match bspec {
                BinSpec::Automatic | BinSpec::LogScaled | BinSpec::FixedNumber(_) =>
                    (gmax - gmin) / (nbins as f64),
                BinSpec::FixedSize(dx) =>
                    dx,
            }
        };

        //println!("{}: number of bins = {}, bin volume = {:e}", rank, nbins, bin_vol);

        // Binning
        let mut cts: Vec<f64> = vec![0.0; nbins];
        let mut total = 0.0;

        for e in base.iter() {
            let value = if bspec == BinSpec::LogScaled {
                accessor(e).ln()
            } else {
                accessor(e)
            };

            let bin = ((value - gmin) / bin_vol).floor() as usize;

            let w = weight(e);
            total = total + w; // count everything, even if not binned

            if !value.is_finite() {
                continue;
            }

            // adjust weight to include actual size of bin / log-scaled size
            let w = if bspec == BinSpec::LogScaled && (hspec == HeightSpec::Density || hspec == HeightSpec::ProbabilityDensity) {
                w * bin_vol / linear_bin_vol(gmin, bin_vol, bin)
            } else {w};

            // access by row-major order
            let fbin = bin;
            if fbin < cts.len() {
                cts[fbin] = cts[fbin] + w;
            }
        }

        // total weight across world
        let mut gtotal = 0.0;
        comm.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
        //println!("{}: local total = {:e}, global = {:e}", rank, total, gtotal);

        let cts = match hspec {
            HeightSpec::Count => cts,
            HeightSpec::Density => cts.iter().map(|ct| ct / bin_vol).collect(),
            HeightSpec::ProbabilityDensity => cts.iter().map(|ct| ct / (bin_vol * gtotal)).collect()
        };

        // All reduce?
        let mut gcts: Vec<f64> = vec![0.0; nbins];
        comm.all_reduce_into(&cts[..], &mut gcts[..], SystemOperation::sum());

        Some(Histogram {
            dim: 1,
            total: gtotal,
            bin_vol: bin_vol,
            min: vec![gmin],
            max: vec![gmax],
            cts: gcts,
            bins: vec![nbins],
            bin_sz: if nbins <= 1 {vec![0.0]} else {vec![bin_vol]},
            name: format!("hgram/{}/{}", hspec, name),
            bunit: format!("1/{}", unit),
            axis: vec![format!("{}", name)],
            unit: vec![unit.to_string()],
        })
    }

    #[allow(unused)]
    pub fn generate_2d<T>(
        comm: &impl Communicator,
        base: &[T], accessor: [&impl Fn(&T) -> f64; 2], weight: &impl Fn(&T) -> f64,
        name: [&str; 2], unit: [&str; 2],
        bspec: [BinSpec; 2], hspec: HeightSpec) -> Option<Histogram> {
        //let rank = comm.rank();

        // Local min and max

        let (xmin, xmax) = if bspec[0] == BinSpec::LogScaled {
            min_max_by(base, accessor[0], f64::ln).unwrap_or((std::f64::MAX, -std::f64::MAX))
        } else {
            min_max_by(base, accessor[0], std::convert::identity).unwrap_or((std::f64::MAX, -std::f64::MAX))
        };

        let (ymin, ymax) = if bspec[1] == BinSpec::LogScaled {
            min_max_by(base, accessor[1], f64::ln).unwrap_or((std::f64::MAX, -std::f64::MAX))
        } else {
            min_max_by(base, accessor[1], std::convert::identity).unwrap_or((std::f64::MAX, -std::f64::MAX))
        };

        let min = [xmin, ymin];
        let max = [xmax, ymax];

        //Global min and max

        let mut gmin = [0.0; 2];
        let mut gmax = [0.0; 2];
        let mut gnum: usize = 0;
        comm.all_reduce_into(&min[..], &mut gmin[..], SystemOperation::min());
        comm.all_reduce_into(&max[..], &mut gmax[..], SystemOperation::max());
        comm.all_reduce_into(&base.len(), &mut gnum, SystemOperation::sum());

        if gnum == 0 {
            return None;
        }
        
        // Prep bins

        let nbins = [
            number_of_bins(gmin[0], gmax[0], gnum, bspec[0]),
            number_of_bins(gmin[1], gmax[1], gnum, bspec[1]),
        ];

        let (bin_sz, bin_vol) = bin_size_and_volume(2, &gmin, &gmax, &nbins, &bspec);

        //println!("{}: number of bins = {:?}, bin volume = {:e}", rank, nbins, bin_vol);

        // Binning
        let mut cts: Vec<f64> = vec![0.0; nbins[0] * nbins[1]];
        let mut total = 0.0;

        for e in base.iter() {
            let value: Vec<f64> = bspec.iter().zip(accessor.iter())
                .map(|(&b, &f)| if b == BinSpec::LogScaled {f(e).ln()} else {f(e)})
                .collect();

            let mut w = weight(e);
            total = total + w; // count everything, even if not binned

            if value.iter().any(|&x| !x.is_finite()) {
                continue; // all of value[i] must be finite
            }

            let bin = [
                if bin_sz[0] == 0.0 {0} else {((value[0] - gmin[0]) / bin_sz[0]).floor() as usize},
                if bin_sz[1] == 0.0 {0} else {((value[1] - gmin[1]) / bin_sz[1]).floor() as usize},
            ];

            // adjust weight to include actual size of bin / log-scaled size
            if bspec[0] == BinSpec::LogScaled && (hspec == HeightSpec::Density || hspec == HeightSpec::ProbabilityDensity) {
                w *= bin_sz[0] / linear_bin_vol(gmin[0], bin_sz[0], bin[0]);
            }

            if bspec[1] == BinSpec::LogScaled && (hspec == HeightSpec::Density || hspec == HeightSpec::ProbabilityDensity) {
                w *= bin_sz[1] / linear_bin_vol(gmin[1], bin_sz[1], bin[1]);
            }

            let fbin = bin[1] * nbins[1] + bin[0]; // row_index * elements_in_row + column_index
            if fbin < cts.len() {
                cts[fbin] = cts[fbin] + w;
            }
        }

        // total weight across world
        let mut gtotal = 0.0;
        comm.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
        //println!("{}: local total = {:e}, global = {:e}", rank, total, gtotal);

        let cts = match hspec {
            HeightSpec::Count => cts,
            HeightSpec::Density => cts.iter().map(|ct| ct / bin_vol).collect(),
            HeightSpec::ProbabilityDensity => cts.iter().map(|ct| ct / (bin_vol * gtotal)).collect()
        };

        // All reduce?
        let mut gcts: Vec<f64> = vec![0.0; nbins[0] * nbins[1]];
        comm.all_reduce_into(&cts[..], &mut gcts[..], SystemOperation::sum());

        Some(Histogram {
            dim: 2,
            total: gtotal,
            bin_vol: bin_vol,
            min: gmin.to_vec(),
            max: gmax.to_vec(),
            cts: gcts,
            bins: nbins.to_vec(),
            bin_sz: bin_sz,
            name: format!("hgram/{}/{}_{}", hspec, name[0], name[1]),
            bunit: format!("1/({}.{})", unit[0], unit[1]),
            axis: vec![name[0].to_owned(), name[1].to_owned()],
            unit: vec![unit[0].to_owned(), unit[1].to_owned()],
        })
    }

    pub fn write_fits(&self, filename: &str) -> Result<(),errors::Error> {
        use fitsio::images::{ImageDescription, ImageType};
        let desc = ImageDescription {
            data_type: ImageType::Double,
            dimensions: &self.bins[..],
        };
        let mut file = FitsFile::create(filename).with_custom_primary(&desc).open()?;
        let hdu = file.hdu(0)?;

        // Write metadata
        for i in 0..self.dim {
            hdu.write_key(&mut file, &format!("CRPIX{}", i+1), 1.0)?; // pixel centre
            hdu.write_key(&mut file, &format!("CRVAL{}", i+1), self.min[i] + 0.5 * self.bin_sz[i])?;
            hdu.write_key(&mut file, &format!("CDELT{}", i+1), self.bin_sz[i])?;
            hdu.write_key(&mut file, &format!("CNAME{}", i+1), &self.axis[i][..])?;
            hdu.write_key(&mut file, &format!("CUNIT{}", i+1), &self.unit[i][..])?;
        }

        hdu.write_key(&mut file, "BUNIT", &self.bunit[..])?;
        hdu.write_key(&mut file, "TOTAL", self.total)?;
        hdu.write_key(&mut file, "OBJECT", &self.name[..])?;

        let min = self.cts.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let max = self.cts.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        hdu.write_key(&mut file, "DATAMIN", *min)?;
        hdu.write_key(&mut file, "DATAMAX", *max)?;

        //Write data
        hdu.write_image(&mut file, &self.cts[..])?;

        Ok(())
    }
}