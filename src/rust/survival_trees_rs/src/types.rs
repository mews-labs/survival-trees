use ndarray::Array2;

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub entry: f64,
    pub time: f64,
    pub event: bool,
}

pub struct Dataset {
    pub x: Array2<f64>,
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }
    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }
}

#[derive(Clone, Debug)]
pub struct KmCurve {
    pub times: Vec<f64>,
    pub surv: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitCriterion {
    LogRank,
    PoissonExp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipeline {
    /// Nelson-Aalen at leaves + CHF averaging across trees. Canonical RSF.
    Aalen,
    /// Kaplan-Meier at leaves + arithmetic mean of S across trees. Breiman.
    Km,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitterMode {
    /// Exhaustive sweep over candidate thresholds (Breiman / Ishwaran).
    Best,
    /// One uniformly random threshold per feature (Geurts 2006).
    Random,
}

#[derive(Clone, Copy, Debug)]
pub struct Control {
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub min_impurity_decrease: f64,
    pub criterion: SplitCriterion,
    pub pipeline: Pipeline,
    pub splitter: SplitterMode,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            min_impurity_decrease: 0.0,
            criterion: SplitCriterion::LogRank,
            pipeline: Pipeline::Aalen,
            splitter: SplitterMode::Best,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub feature: usize,
    pub threshold: f64,
    pub score: f64,
    /// Direction followed by NaN `X[f]` at this node — MIA (Twala 2008).
    pub default_direction: Direction,
}

pub enum Node {
    Internal {
        feature: usize,
        threshold: f64,
        default_direction: Direction,
        left: Box<Node>,
        right: Box<Node>,
        n_samples: usize,
        improvement: f64,
    },
    Leaf {
        leaf_id: usize,
        km: KmCurve,
    },
}

pub struct Tree {
    pub root: Node,
    pub n_leaves: usize,
    pub n_features: usize,
    pub pipeline: Pipeline,
}

pub struct Forest {
    pub trees: Vec<Tree>,
    pub feature_subsets: Vec<Vec<usize>>,
    pub n_features: usize,
    pub pipeline: Pipeline,
}
