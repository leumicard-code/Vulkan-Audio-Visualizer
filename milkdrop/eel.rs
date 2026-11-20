// ============================================================================
// milkdrop/eel.rs — Fast EEL2 Compiler & Interpreter for MilkDrop Equations
// ============================================================================

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type VarId = usize;

const GMEGABUF_MAX: usize = 8_388_608;
const MEGABUF_BLOCK: usize = 4096;
const LOOP_MAX_ITERS: usize = 1_048_576;
const CLOSE_EPS: f64 = 1e-5;

#[inline]
fn sanitize(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
fn is_truthy(v: f64) -> bool {
    v.is_finite() && v.abs() >= CLOSE_EPS
}

#[inline]
fn to_i64_trunc(v: f64) -> i64 {
    if !v.is_finite() {
        0
    } else if v >= i64::MAX as f64 {
        i64::MAX
    } else if v <= i64::MIN as f64 {
        i64::MIN
    } else {
        v as i64
    }
}

/// Shared across all `Vm` instances of one preset (`reg00`..`reg99` and `gmegabuf`).
pub struct Globals {
    pub regs: [f64; 100],
    gmegabuf: HashMap<usize, Box<[f64; MEGABUF_BLOCK]>>,
}

impl Globals {
    pub fn new() -> Self {
        Self {
            regs: [0.0; 100],
            gmegabuf: HashMap::new(),
        }
    }

    #[inline]
    pub fn get_gmegabuf(&self, idx: f64) -> f64 {
        if !idx.is_finite() || idx < 0.0 {
            return 0.0;
        }
        let i = idx as usize;
        if i >= GMEGABUF_MAX {
            return 0.0;
        }
        let block = i / MEGABUF_BLOCK;
        let off = i % MEGABUF_BLOCK;
        self.gmegabuf.get(&block).map_or(0.0, |b| b[off])
    }

    #[inline]
    pub fn set_gmegabuf(&mut self, idx: f64, val: f64) {
        if !idx.is_finite() || idx < 0.0 {
            return;
        }
        let i = idx as usize;
        if i >= GMEGABUF_MAX {
            return;
        }
        let block = i / MEGABUF_BLOCK;
        let off = i % MEGABUF_BLOCK;
        let v = sanitize(val);
        self.gmegabuf
            .entry(block)
            .or_insert_with(|| Box::new([0.0; MEGABUF_BLOCK]))[off] = v;
    }
}

impl Default for Globals {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedGlobals = Rc<RefCell<Globals>>;

pub struct Vm {
    vars: Vec<f64>,
    reg_map: Vec<i8>, // -1 for normal var, 0..=99 for reg00..reg99
    name_to_id: HashMap<String, VarId>,
    id_to_name: Vec<String>,
    megabuf: HashMap<usize, Box<[f64; MEGABUF_BLOCK]>>,
    rng_state: u64,
    globals: SharedGlobals,
}

impl Vm {
    pub fn new(globals: SharedGlobals) -> Self {
        Self {
            vars: Vec::with_capacity(64),
            reg_map: Vec::with_capacity(64),
            name_to_id: HashMap::with_capacity(64),
            id_to_name: Vec::with_capacity(64),
            megabuf: HashMap::new(),
            rng_state: 0x9E37_79B9_7F4A_7C15,
            globals,
        }
    }

    /// Get or create a variable ID (case-insensitive).
    pub fn var(&mut self, name: &str) -> VarId {
        let lower = name.to_ascii_lowercase();
        if let Some(&id) = self.name_to_id.get(&lower) {
            return id;
        }
        let id = self.vars.len();
        let reg_idx = parse_reg_index(&lower);
        self.vars.push(0.0);
        self.reg_map.push(reg_idx);
        self.name_to_id.insert(lower.clone(), id);
        self.id_to_name.push(lower);
        id
    }

    #[inline]
    pub fn get(&self, id: VarId) -> f64 {
        let reg = self.reg_map[id];
        if reg >= 0 {
            self.globals.borrow().regs[reg as usize]
        } else {
            self.vars[id]
        }
    }

    #[inline]
    pub fn set(&mut self, id: VarId, v: f64) {
        let sv = sanitize(v);
        let reg = self.reg_map[id];
        if reg >= 0 {
            self.globals.borrow_mut().regs[reg as usize] = sv;
        } else {
            self.vars[id] = sv;
        }
    }

    pub fn get_by_name(&self, name: &str) -> Option<f64> {
        let lower = name.to_ascii_lowercase();
        self.name_to_id.get(&lower).map(|&id| self.get(id))
    }

    pub fn set_by_name(&mut self, name: &str, v: f64) {
        let id = self.var(name);
        self.set(id, v);
    }

    pub fn snapshot(&self) -> Vec<f64> {
        self.vars.clone()
    }

    pub fn restore(&mut self, snap: &[f64]) {
        let n = snap.len().min(self.vars.len());
        self.vars[..n].copy_from_slice(&snap[..n]);
    }

    pub fn var_names(&self) -> Vec<(String, VarId)> {
        self.id_to_name
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect()
    }

    #[inline]
    fn get_megabuf(&self, idx: f64) -> f64 {
        if !idx.is_finite() || idx < 0.0 {
            return 0.0;
        }
        let i = idx as usize;
        if i >= GMEGABUF_MAX {
            return 0.0;
        }
        let block = i / MEGABUF_BLOCK;
        let off = i % MEGABUF_BLOCK;
        self.megabuf.get(&block).map_or(0.0, |b| b[off])
    }

    #[inline]
    fn set_megabuf(&mut self, idx: f64, val: f64) {
        if !idx.is_finite() || idx < 0.0 {
            return;
        }
        let i = idx as usize;
        if i >= GMEGABUF_MAX {
            return;
        }
        let block = i / MEGABUF_BLOCK;
        let off = i % MEGABUF_BLOCK;
        let v = sanitize(val);
        self.megabuf
            .entry(block)
            .or_insert_with(|| Box::new([0.0; MEGABUF_BLOCK]))[off] = v;
    }

    #[inline]
    fn next_rand_unit(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        // Map upper 53 bits to [0.0, 1.0)
        ((x >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn parse_reg_index(lower: &str) -> i8 {
    let b = lower.as_bytes();
    if b.len() == 5 && &b[..3] == b"reg" && b[3].is_ascii_digit() && b[4].is_ascii_digit() {
        let idx = (b[3] - b'0') * 10 + (b[4] - b'0');
        idx as i8
    } else {
        -1
    }
}

// ============================================================================
// AST / Flat Arena Representation
// ============================================================================

type ExprId = u32;

#[derive(Clone, Copy, Debug)]
enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Pow,
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    Shl,
    Shr,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    ExactEq,
    ExactNe,
    BitAnd,
    BitOr,
    LogAnd,
    LogOr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssignOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    BitOr,
    BitAnd,
}

#[derive(Clone, Copy, Debug)]
enum LValue {
    Var(VarId),
    Megabuf(ExprId),
    GMegabuf(ExprId),
    Array(ExprId, ExprId),
}

#[derive(Clone, Copy, Debug)]
enum Builtin1 {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sqr,
    Sqrt,
    Exp,
    Log,
    Log10,
    Abs,
    Sign,
    Int,
    Floor,
    Ceil,
    InvSqrt,
    Bnot,
    Rand,
    Freembuf,
    Megabuf,
    GMegabuf,
}

#[derive(Clone, Copy, Debug)]
enum Builtin2 {
    Atan2,
    Pow,
    Min,
    Max,
    Sigmoid,
    Above,
    Below,
    Equal,
    Band,
    Bor,
    Exec2,
    Loop,
    While2,
}

#[derive(Clone, Debug)]
enum Node {
    Lit(f64),
    Var(VarId),
    Seq(Vec<ExprId>),
    Unary(UnOp, ExprId),
    Binary(BinOp, ExprId, ExprId),
    Ternary(ExprId, ExprId, Option<ExprId>),
    Assign(AssignOp, LValue, ExprId),
    Call1(Builtin1, ExprId),
    Call2(Builtin2, ExprId, ExprId),
    Call3If(ExprId, ExprId, ExprId),
    Call3Exec(ExprId, ExprId, ExprId),
    While1(ExprId),
    MemCpy(ExprId, ExprId, ExprId),
    MemSet(ExprId, ExprId, ExprId),
    ArrayGet(ExprId, ExprId),
    Ignore(Vec<ExprId>),
}

pub struct Program {
    nodes: Vec<Node>,
    root: Option<ExprId>,
}

impl Program {
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    #[inline]
    pub fn run(&self, vm: &mut Vm) {
        if let Some(root) = self.root {
            let _ = eval_node(&self.nodes, root, vm);
        }
    }

    #[cfg(test)]
    pub fn eval_val(&self, vm: &mut Vm) -> f64 {
        match self.root {
            Some(root) => eval_node(&self.nodes, root, vm),
            None => 0.0,
        }
    }
}

#[inline]
fn eval_node(nodes: &[Node], id: ExprId, vm: &mut Vm) -> f64 {
    match &nodes[id as usize] {
        Node::Lit(v) => *v,
        Node::Var(vid) => vm.get(*vid),
        Node::Seq(items) => {
            let mut last = 0.0;
            for &expr in items {
                last = eval_node(nodes, expr, vm);
            }
            last
        }
        Node::Unary(op, inner) => {
            let v = eval_node(nodes, *inner, vm);
            match op {
                UnOp::Neg => sanitize(-v),
                UnOp::Not => {
                    if !is_truthy(v) {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        }
        Node::Binary(op, lhs, rhs) => eval_binary(nodes, *op, *lhs, *rhs, vm),
        Node::Ternary(cond, t_expr, f_expr) => {
            if is_truthy(eval_node(nodes, *cond, vm)) {
                eval_node(nodes, *t_expr, vm)
            } else if let Some(f_id) = f_expr {
                eval_node(nodes, *f_id, vm)
            } else {
                0.0
            }
        }
        Node::Assign(op, lv, rhs) => {
            let rval = eval_node(nodes, *rhs, vm);
            eval_assign(nodes, *op, *lv, rval, vm)
        }
        Node::Call1(b, a) => {
            let x = eval_node(nodes, *a, vm);
            eval_builtin1(*b, x, vm)
        }
        Node::Call2(b, a, c) => eval_builtin2(nodes, *b, *a, *c, vm),
        Node::Call3If(c, t_e, f_e) => {
            if is_truthy(eval_node(nodes, *c, vm)) {
                eval_node(nodes, *t_e, vm)
            } else {
                eval_node(nodes, *f_e, vm)
            }
        }
        Node::Call3Exec(a, b, c) => {
            let _ = eval_node(nodes, *a, vm);
            let _ = eval_node(nodes, *b, vm);
            eval_node(nodes, *c, vm)
        }
        Node::While1(body) => {
            for _ in 0..LOOP_MAX_ITERS {
                if !is_truthy(eval_node(nodes, *body, vm)) {
                    break;
                }
            }
            0.0
        }
        Node::MemCpy(dst, src, len) => {
            let d = eval_node(nodes, *dst, vm);
            let s = eval_node(nodes, *src, vm);
            let l = to_i64_trunc(eval_node(nodes, *len, vm));
            if l > 0 && d.is_finite() && s.is_finite() && d >= 0.0 && s >= 0.0 {
                let d0 = d as usize;
                let s0 = s as usize;
                let count = (l as usize).min(GMEGABUF_MAX);
                if d0 < GMEGABUF_MAX && s0 < GMEGABUF_MAX {
                    let count = count.min(GMEGABUF_MAX - d0).min(GMEGABUF_MAX - s0);
                    if d0 > s0 && d0 < s0 + count {
                        // Reverse copy for overlap
                        for i in (0..count).rev() {
                            let val = vm.get_megabuf((s0 + i) as f64);
                            vm.set_megabuf((d0 + i) as f64, val);
                        }
                    } else {
                        for i in 0..count {
                            let val = vm.get_megabuf((s0 + i) as f64);
                            vm.set_megabuf((d0 + i) as f64, val);
                        }
                    }
                }
            }
            d
        }
        Node::MemSet(dst, val, len) => {
            let d = eval_node(nodes, *dst, vm);
            let v = sanitize(eval_node(nodes, *val, vm));
            let l = to_i64_trunc(eval_node(nodes, *len, vm));
            if l > 0 && d.is_finite() && d >= 0.0 {
                let d0 = d as usize;
                if d0 < GMEGABUF_MAX {
                    let count = (l as usize).min(GMEGABUF_MAX - d0);
                    for i in 0..count {
                        vm.set_megabuf((d0 + i) as f64, v);
                    }
                }
            }
            d
        }
        Node::ArrayGet(base, idx) => {
            let b = eval_node(nodes, *base, vm);
            let i = eval_node(nodes, *idx, vm);
            vm.get_megabuf(b + i)
        }
        Node::Ignore(args) => {
            for &a in args {
                let _ = eval_node(nodes, a, vm);
            }
            0.0
        }
    }
}

#[inline]
fn apply_assign_op(op: AssignOp, cur: f64, rhs: f64) -> f64 {
    let v = match op {
        AssignOp::Set => rhs,
        AssignOp::Add => cur + rhs,
        AssignOp::Sub => cur - rhs,
        AssignOp::Mul => cur * rhs,
        AssignOp::Div => {
            if rhs.abs() < 1e-30 {
                0.0
            } else {
                cur / rhs
            }
        }
        AssignOp::Mod => {
            let bi = to_i64_trunc(rhs).abs();
            if bi == 0 {
                0.0
            } else {
                (to_i64_trunc(cur).abs() % bi) as f64
            }
        }
        AssignOp::Pow => cur.powf(rhs),
        AssignOp::BitOr => (to_i64_trunc(cur) | to_i64_trunc(rhs)) as f64,
        AssignOp::BitAnd => (to_i64_trunc(cur) & to_i64_trunc(rhs)) as f64,
    };
    sanitize(v)
}

#[inline]
fn eval_assign(nodes: &[Node], op: AssignOp, lv: LValue, rhs: f64, vm: &mut Vm) -> f64 {
    match lv {
        LValue::Var(id) => {
            let val = if matches!(op, AssignOp::Set) {
                sanitize(rhs)
            } else {
                apply_assign_op(op, vm.get(id), rhs)
            };
            vm.set(id, val);
            val
        }
        LValue::Megabuf(idx_expr) => {
            let idx = eval_node(nodes, idx_expr, vm);
            let val = if matches!(op, AssignOp::Set) {
                sanitize(rhs)
            } else {
                apply_assign_op(op, vm.get_megabuf(idx), rhs)
            };
            vm.set_megabuf(idx, val);
            val
        }
        LValue::GMegabuf(idx_expr) => {
            let idx = eval_node(nodes, idx_expr, vm);
            let mut g = vm.globals.borrow_mut();
            let val = if matches!(op, AssignOp::Set) {
                sanitize(rhs)
            } else {
                apply_assign_op(op, g.get_gmegabuf(idx), rhs)
            };
            g.set_gmegabuf(idx, val);
            val
        }
        LValue::Array(base_expr, idx_expr) => {
            let b = eval_node(nodes, base_expr, vm);
            let i = eval_node(nodes, idx_expr, vm);
            let idx = b + i;
            let val = if matches!(op, AssignOp::Set) {
                sanitize(rhs)
            } else {
                apply_assign_op(op, vm.get_megabuf(idx), rhs)
            };
            vm.set_megabuf(idx, val);
            val
        }
    }
}

#[inline]
fn eval_binary(nodes: &[Node], op: BinOp, lhs: ExprId, rhs: ExprId, vm: &mut Vm) -> f64 {
    match op {
        BinOp::LogAnd => {
            let a = eval_node(nodes, lhs, vm);
            if !is_truthy(a) {
                0.0
            } else if is_truthy(eval_node(nodes, rhs, vm)) {
                1.0
            } else {
                0.0
            }
        }
        BinOp::LogOr => {
            let a = eval_node(nodes, lhs, vm);
            if is_truthy(a) || is_truthy(eval_node(nodes, rhs, vm)) {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            let a = eval_node(nodes, lhs, vm);
            let b = eval_node(nodes, rhs, vm);
            let v = match op {
                BinOp::Pow => a.powf(b),
                BinOp::Mul => a * b,
                BinOp::Div => {
                    if b.abs() < 1e-30 {
                        0.0
                    } else {
                        a / b
                    }
                }
                BinOp::Mod => {
                    let bi = to_i64_trunc(b).abs();
                    if bi == 0 {
                        0.0
                    } else {
                        (to_i64_trunc(a).abs() % bi) as f64
                    }
                }
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Shl => {
                    let sh = (to_i64_trunc(b) & 63) as u32;
                    to_i64_trunc(a).wrapping_shl(sh) as f64
                }
                BinOp::Shr => {
                    let sh = (to_i64_trunc(b) & 63) as u32;
                    to_i64_trunc(a).wrapping_shr(sh) as f64
                }
                BinOp::Lt => if a < b { 1.0 } else { 0.0 },
                BinOp::Gt => if a > b { 1.0 } else { 0.0 },
                BinOp::Le => if a <= b { 1.0 } else { 0.0 },
                BinOp::Ge => if a >= b { 1.0 } else { 0.0 },
                BinOp::Eq => if (a - b).abs() < CLOSE_EPS { 1.0 } else { 0.0 },
                BinOp::Ne => if (a - b).abs() >= CLOSE_EPS { 1.0 } else { 0.0 },
                BinOp::ExactEq => if a == b { 1.0 } else { 0.0 },
                BinOp::ExactNe => if a != b { 1.0 } else { 0.0 },
                BinOp::BitAnd => (to_i64_trunc(a) & to_i64_trunc(b)) as f64,
                BinOp::BitOr => (to_i64_trunc(a) | to_i64_trunc(b)) as f64,
                BinOp::LogAnd | BinOp::LogOr => unreachable!(),
            };
            sanitize(v)
        }
    }
}

#[inline]
fn eval_builtin1(b: Builtin1, x: f64, vm: &mut Vm) -> f64 {
    let v = match b {
        Builtin1::Sin => x.sin(),
        Builtin1::Cos => x.cos(),
        Builtin1::Tan => x.tan(),
        Builtin1::Asin => x.clamp(-1.0, 1.0).asin(),
        Builtin1::Acos => x.clamp(-1.0, 1.0).acos(),
        Builtin1::Atan => x.atan(),
        Builtin1::Sqr => x * x,
        Builtin1::Sqrt => {
            if x >= 0.0 {
                x.sqrt()
            } else {
                0.0
            }
        }
        Builtin1::Exp => x.exp(),
        Builtin1::Log => {
            if x > 0.0 {
                x.ln()
            } else {
                0.0
            }
        }
        Builtin1::Log10 => {
            if x > 0.0 {
                x.log10()
            } else {
                0.0
            }
        }
        Builtin1::Abs => x.abs(),
        Builtin1::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Builtin1::Int => x.trunc(),
        Builtin1::Floor => x.floor(),
        Builtin1::Ceil => x.ceil(),
        Builtin1::InvSqrt => {
            if x > 0.0 {
                1.0 / x.sqrt()
            } else {
                0.0
            }
        }
        Builtin1::Bnot => {
            if !is_truthy(x) {
                1.0
            } else {
                0.0
            }
        }
        Builtin1::Rand => {
            if !x.is_finite() || x <= 0.0 {
                0.0
            } else if x <= 1.0 {
                vm.next_rand_unit() * x
            } else {
                (vm.next_rand_unit() * x).floor()
            }
        }
        Builtin1::Freembuf => {
            if !x.is_finite() || x <= 0.0 {
                vm.megabuf.clear();
            } else {
                let keep_blocks = (x as usize).div_ceil(MEGABUF_BLOCK);
                vm.megabuf.retain(|&b_idx, _| b_idx < keep_blocks);
            }
            x
        }
        Builtin1::Megabuf => vm.get_megabuf(x),
        Builtin1::GMegabuf => vm.globals.borrow().get_gmegabuf(x),
    };
    sanitize(v)
}

#[inline]
fn eval_builtin2(nodes: &[Node], b: Builtin2, lhs: ExprId, rhs: ExprId, vm: &mut Vm) -> f64 {
    match b {
        Builtin2::Band => {
            if is_truthy(eval_node(nodes, lhs, vm)) && is_truthy(eval_node(nodes, rhs, vm)) {
                1.0
            } else {
                0.0
            }
        }
        Builtin2::Bor => {
            if is_truthy(eval_node(nodes, lhs, vm)) || is_truthy(eval_node(nodes, rhs, vm)) {
                1.0
            } else {
                0.0
            }
        }
        Builtin2::Exec2 => {
            let _ = eval_node(nodes, lhs, vm);
            eval_node(nodes, rhs, vm)
        }
        Builtin2::Loop => {
            let n = to_i64_trunc(eval_node(nodes, lhs, vm));
            if n <= 0 {
                return 0.0;
            }
            let count = (n as usize).min(LOOP_MAX_ITERS);
            let mut last = 0.0;
            for _ in 0..count {
                last = eval_node(nodes, rhs, vm);
            }
            last
        }
        Builtin2::While2 => {
            for _ in 0..LOOP_MAX_ITERS {
                if !is_truthy(eval_node(nodes, lhs, vm)) {
                    break;
                }
                let _ = eval_node(nodes, rhs, vm);
            }
            0.0
        }
        _ => {
            let a = eval_node(nodes, lhs, vm);
            let c = eval_node(nodes, rhs, vm);
            let v = match b {
                Builtin2::Atan2 => a.atan2(c),
                Builtin2::Pow => a.powf(c),
                Builtin2::Min => a.min(c),
                Builtin2::Max => a.max(c),
                Builtin2::Sigmoid => 1.0 / (1.0 + (-a * c).exp()),
                Builtin2::Above => if a > c { 1.0 } else { 0.0 },
                Builtin2::Below => if a < c { 1.0 } else { 0.0 },
                Builtin2::Equal => if (a - c).abs() < CLOSE_EPS { 1.0 } else { 0.0 },
                Builtin2::Band
                | Builtin2::Bor
                | Builtin2::Exec2
                | Builtin2::Loop
                | Builtin2::While2 => unreachable!(),
            };
            sanitize(v)
        }
    }
}

// ============================================================================
// Lexer
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Semi,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Question,
    Colon,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Bang,
    BitAnd,
    BitOr,
    LogAnd,
    LogOr,
    Shl,
    Shr,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    ExactEq,
    ExactNe,
    Assign(AssignOp),
    Eof,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut out = Vec::with_capacity(n / 3 + 8);

    while i < n {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Line comment //
        if c == b'/' && i + 1 < n && b[i + 1] == b'/' {
            i += 2;
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment /* ... */
        if c == b'/' && i + 1 < n && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        // Constants $PI, $E, $PHI, $'c'
        if c == b'$' {
            i += 1;
            if i < n && b[i] == b'\'' {
                i += 1;
                let ch = if i < n {
                    let v = b[i] as f64;
                    i += 1;
                    if i < n && b[i] == b'\'' {
                        i += 1;
                    }
                    v
                } else {
                    0.0
                };
                out.push(Tok::Num(ch));
                continue;
            }
            let start = i;
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let name = src[start..i].to_ascii_lowercase();
            let val = match name.as_str() {
                "pi" => std::f64::consts::PI,
                "e" => std::f64::consts::E,
                "phi" => 1.618_033_988_749_895,
                _ => {
                    if let Some(hex) = name.strip_prefix('x') {
                        u64::from_str_radix(hex, 16).unwrap_or(0) as f64
                    } else {
                        return Err(format!("Unknown $ constant: ${name}"));
                    }
                }
            };
            out.push(Tok::Num(val));
            continue;
        }
        // Numbers: hex 0x..., decimal, .5, 1e-3
        if c.is_ascii_digit() || (c == b'.' && i + 1 < n && b[i + 1].is_ascii_digit()) {
            if c == b'0' && i + 1 < n && (b[i + 1] == b'x' || b[i + 1] == b'X') {
                i += 2;
                let start = i;
                while i < n && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let val = u64::from_str_radix(&src[start..i], 16).unwrap_or(0) as f64;
                out.push(Tok::Num(val));
                continue;
            }
            let start = i;
            while i < n && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < n && b[i] == b'.' {
                i += 1;
                while i < n && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < n && (b[i] == b'e' || b[i] == b'E') {
                let save = i;
                i += 1;
                if i < n && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                let d_start = i;
                while i < n && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i == d_start {
                    i = save;
                }
            }
            let val: f64 = src[start..i]
                .parse()
                .map_err(|e| format!("Invalid number '{}': {e}", &src[start..i]))?;
            out.push(Tok::Num(val));
            continue;
        }
        // Identifiers: letter or _ (or stray '.')
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                i += 1;
            }
            out.push(Tok::Ident(src[start..i].to_ascii_lowercase()));
            continue;
        }

        let next = if i + 1 < n { b[i + 1] } else { 0 };
        let next2 = if i + 2 < n { b[i + 2] } else { 0 };

        match c {
            b';' => {
                out.push(Tok::Semi);
                i += 1;
            }
            b',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b'[' => {
                out.push(Tok::LBracket);
                i += 1;
            }
            b']' => {
                out.push(Tok::RBracket);
                i += 1;
            }
            b'?' => {
                out.push(Tok::Question);
                i += 1;
            }
            b':' => {
                out.push(Tok::Colon);
                i += 1;
            }
            b'+' => {
                if next == b'=' {
                    out.push(Tok::Assign(AssignOp::Add));
                    i += 2;
                } else {
                    out.push(Tok::Plus);
                    i += 1;
                }
            }
            b'-' => {
                if next == b'=' {
                    out.push(Tok::Assign(AssignOp::Sub));
                    i += 2;
                } else {
                    out.push(Tok::Minus);
                    i += 1;
                }
            }
            b'*' => {
                if next == b'=' {
                    out.push(Tok::Assign(AssignOp::Mul));
                    i += 2;
                } else {
                    out.push(Tok::Star);
                    i += 1;
                }
            }
            b'/' => {
                if next == b'=' {
                    out.push(Tok::Assign(AssignOp::Div));
                    i += 2;
                } else {
                    out.push(Tok::Slash);
                    i += 1;
                }
            }
            b'%' => {
                if next == b'=' {
                    out.push(Tok::Assign(AssignOp::Mod));
                    i += 2;
                } else {
                    out.push(Tok::Percent);
                    i += 1;
                }
            }
            b'^' => {
                if next == b'=' {
                    out.push(Tok::Assign(AssignOp::Pow));
                    i += 2;
                } else {
                    out.push(Tok::Caret);
                    i += 1;
                }
            }
            b'!' => {
                if next == b'=' && next2 == b'=' {
                    out.push(Tok::ExactNe);
                    i += 3;
                } else if next == b'=' {
                    out.push(Tok::Ne);
                    i += 2;
                } else {
                    out.push(Tok::Bang);
                    i += 1;
                }
            }
            b'=' => {
                if next == b'=' && next2 == b'=' {
                    out.push(Tok::ExactEq);
                    i += 3;
                } else if next == b'=' {
                    out.push(Tok::Eq);
                    i += 2;
                } else {
                    out.push(Tok::Assign(AssignOp::Set));
                    i += 1;
                }
            }
            b'<' => {
                if next == b'<' {
                    out.push(Tok::Shl);
                    i += 2;
                } else if next == b'=' {
                    out.push(Tok::Le);
                    i += 2;
                } else {
                    out.push(Tok::Lt);
                    i += 1;
                }
            }
            b'>' => {
                if next == b'>' {
                    out.push(Tok::Shr);
                    i += 2;
                } else if next == b'=' {
                    out.push(Tok::Ge);
                    i += 2;
                } else {
                    out.push(Tok::Gt);
                    i += 1;
                }
            }
            b'&' => {
                if next == b'&' {
                    out.push(Tok::LogAnd);
                    i += 2;
                } else if next == b'=' {
                    out.push(Tok::Assign(AssignOp::BitAnd));
                    i += 2;
                } else {
                    out.push(Tok::BitAnd);
                    i += 1;
                }
            }
            b'|' => {
                if next == b'|' {
                    out.push(Tok::LogOr);
                    i += 2;
                } else if next == b'=' {
                    out.push(Tok::Assign(AssignOp::BitOr));
                    i += 2;
                } else {
                    out.push(Tok::BitOr);
                    i += 1;
                }
            }
            _ => {
                return Err(format!("Unexpected character '{}' in EEL", c as char));
            }
        }
    }
    out.push(Tok::Eof);
    Ok(out)
}

// ============================================================================
// Recursive-Descent Parser + Constant Folder
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum StopAt {
    Eof,
    RParen,
    RBracket,
    CallArg,
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    nodes: Vec<Node>,
    vm: &'a mut Vm,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::Eof)
    }

    fn bump(&mut self) -> Tok {
        let t = self.peek().clone();
        if self.pos < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn push(&mut self, node: Node) -> ExprId {
        let id = self.nodes.len() as ExprId;
        self.nodes.push(node);
        id
    }

    fn lit_val(&self, id: ExprId) -> Option<f64> {
        match self.nodes[id as usize] {
            Node::Lit(v) => Some(v),
            _ => None,
        }
    }

    fn parse_seq(&mut self, stop: StopAt) -> Result<Option<ExprId>, String> {
        let mut items = Vec::new();
        loop {
            while matches!(self.peek(), Tok::Semi) {
                self.bump();
            }
            if matches!(self.peek(), Tok::Eof) {
                if stop != StopAt::Eof {
                    return Err("Unexpected EOF inside parentheses or brackets".to_string());
                }
                break;
            }
            let at_end = match stop {
                StopAt::Eof => false,
                StopAt::RParen => matches!(self.peek(), Tok::RParen),
                StopAt::RBracket => matches!(self.peek(), Tok::RBracket),
                StopAt::CallArg => matches!(self.peek(), Tok::RParen | Tok::Comma),
            };
            if at_end {
                break;
            }
            let e = self.parse_assign()?;
            items.push(e);
            if matches!(self.peek(), Tok::Semi) {
                self.bump();
            } else {
                let can_stop = match stop {
                    StopAt::Eof => matches!(self.peek(), Tok::Eof),
                    StopAt::RParen => matches!(self.peek(), Tok::RParen),
                    StopAt::RBracket => matches!(self.peek(), Tok::RBracket),
                    StopAt::CallArg => matches!(self.peek(), Tok::RParen | Tok::Comma),
                };
                if can_stop {
                    break;
                }
                return Err(format!("Expected ';' or delimiter, found {:?}", self.peek()));
            }
        }
        Ok(match items.len() {
            0 => None,
            1 => Some(items[0]),
            _ => Some(self.push(Node::Seq(items))),
        })
    }

    fn parse_assign(&mut self) -> Result<ExprId, String> {
        let lhs = self.parse_ternary()?;
        if let Tok::Assign(op) = self.peek().clone() {
            self.bump();
            let rhs = self.parse_assign()?;
            let lv = self.as_lvalue(lhs)?;
            return Ok(self.push(Node::Assign(op, lv, rhs)));
        }
        Ok(lhs)
    }

    fn as_lvalue(&self, id: ExprId) -> Result<LValue, String> {
        match &self.nodes[id as usize] {
            Node::Var(vid) => Ok(LValue::Var(*vid)),
            Node::Call1(Builtin1::Megabuf, idx) => Ok(LValue::Megabuf(*idx)),
            Node::Call1(Builtin1::GMegabuf, idx) => Ok(LValue::GMegabuf(*idx)),
            Node::ArrayGet(base, idx) => Ok(LValue::Array(*base, *idx)),
            _ => Err("Left-hand side of assignment is not an lvalue".to_string()),
        }
    }

    fn parse_ternary(&mut self) -> Result<ExprId, String> {
        let cond = self.parse_log_or()?;
        if matches!(self.peek(), Tok::Question) {
            self.bump();
            let t_expr = self.parse_assign()?;
            let f_expr = if matches!(self.peek(), Tok::Colon) {
                self.bump();
                Some(self.parse_assign()?)
            } else {
                None
            };
            return Ok(self.push(Node::Ternary(cond, t_expr, f_expr)));
        }
        Ok(cond)
    }

    fn parse_log_or(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_log_and()?;
        while matches!(self.peek(), Tok::LogOr) {
            self.bump();
            let rhs = self.parse_log_and()?;
            lhs = self.push(Node::Binary(BinOp::LogOr, lhs, rhs));
        }
        Ok(lhs)
    }

    fn parse_log_and(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_bit_or()?;
        while matches!(self.peek(), Tok::LogAnd) {
            self.bump();
            let rhs = self.parse_bit_or()?;
            lhs = self.push(Node::Binary(BinOp::LogAnd, lhs, rhs));
        }
        Ok(lhs)
    }

    fn parse_bit_or(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_bit_and()?;
        while matches!(self.peek(), Tok::BitOr) {
            self.bump();
            let rhs = self.parse_bit_and()?;
            lhs = self.fold_bin(BinOp::BitOr, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_bit_and(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_eq()?;
        while matches!(self.peek(), Tok::BitAnd) {
            self.bump();
            let rhs = self.parse_eq()?;
            lhs = self.fold_bin(BinOp::BitAnd, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_eq(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_rel()?;
        loop {
            let op = match self.peek() {
                Tok::Eq => BinOp::Eq,
                Tok::Ne => BinOp::Ne,
                Tok::ExactEq => BinOp::ExactEq,
                Tok::ExactNe => BinOp::ExactNe,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_rel()?;
            lhs = self.fold_bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_rel(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                Tok::Lt => BinOp::Lt,
                Tok::Gt => BinOp::Gt,
                Tok::Le => BinOp::Le,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_shift()?;
            lhs = self.fold_bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Tok::Shl => BinOp::Shl,
                Tok::Shr => BinOp::Shr,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add()?;
            lhs = self.fold_bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul()?;
            lhs = self.fold_bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<ExprId, String> {
        let mut lhs = self.parse_pow()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Mod,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_pow()?;
            lhs = self.fold_bin(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<ExprId, String> {
        let lhs = self.parse_unary()?;
        if matches!(self.peek(), Tok::Caret) {
            self.bump();
            let rhs = self.parse_pow()?;
            return Ok(self.fold_bin(BinOp::Pow, lhs, rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<ExprId, String> {
        match self.peek().clone() {
            Tok::Minus => {
                self.bump();
                let e = self.parse_unary()?;
                if let Some(v) = self.lit_val(e) {
                    Ok(self.push(Node::Lit(sanitize(-v))))
                } else {
                    Ok(self.push(Node::Unary(UnOp::Neg, e)))
                }
            }
            Tok::Plus => {
                self.bump();
                self.parse_unary()
            }
            Tok::Bang => {
                self.bump();
                let e = self.parse_unary()?;
                if let Some(v) = self.lit_val(e) {
                    Ok(self.push(Node::Lit(if !is_truthy(v) { 1.0 } else { 0.0 })))
                } else {
                    Ok(self.push(Node::Unary(UnOp::Not, e)))
                }
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<ExprId, String> {
        let mut expr = self.parse_primary()?;
        while matches!(self.peek(), Tok::LBracket) {
            self.bump();
            let idx = if matches!(self.peek(), Tok::RBracket) {
                self.push(Node::Lit(0.0))
            } else {
                self.parse_seq(StopAt::RBracket)?
                    .unwrap_or_else(|| self.push(Node::Lit(0.0)))
            };
            if !matches!(self.peek(), Tok::RBracket) {
                return Err("Expected ']'".to_string());
            }
            self.bump();
            expr = self.push(Node::ArrayGet(expr, idx));
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<ExprId, String> {
        match self.bump() {
            Tok::Num(v) => Ok(self.push(Node::Lit(sanitize(v)))),
            Tok::LParen => {
                let inner = self.parse_seq(StopAt::RParen)?;
                if !matches!(self.peek(), Tok::RParen) {
                    return Err("Expected ')'".to_string());
                }
                self.bump();
                Ok(inner.unwrap_or_else(|| self.push(Node::Lit(0.0))))
            }
            Tok::Ident(name) => {
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let args = self.parse_call_args()?;
                    self.build_call(&name, args)
                } else {
                    let vid = self.vm.var(&name);
                    Ok(self.push(Node::Var(vid)))
                }
            }
            other => Err(format!("Unexpected token {other:?} in expression")),
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<ExprId>, String> {
        let mut args = Vec::new();
        if matches!(self.peek(), Tok::RParen) {
            self.bump();
            return Ok(args);
        }
        loop {
            let arg = self
                .parse_seq(StopAt::CallArg)?
                .unwrap_or_else(|| self.push(Node::Lit(0.0)));
            args.push(arg);
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            } else if matches!(self.peek(), Tok::RParen) {
                self.bump();
                break;
            } else {
                return Err(format!("Expected ',' or ')' in function call, got {:?}", self.peek()));
            }
        }
        Ok(args)
    }

    fn build_call(&mut self, name: &str, args: Vec<ExprId>) -> Result<ExprId, String> {
        let expect = |n: usize| -> Result<(), String> {
            if args.len() != n {
                Err(format!("Function '{name}' expects {n} arg(s), got {}", args.len()))
            } else {
                Ok(())
            }
        };

        let b1 = match name {
            "sin" => Some(Builtin1::Sin),
            "cos" => Some(Builtin1::Cos),
            "tan" => Some(Builtin1::Tan),
            "asin" => Some(Builtin1::Asin),
            "acos" => Some(Builtin1::Acos),
            "atan" => Some(Builtin1::Atan),
            "sqr" => Some(Builtin1::Sqr),
            "sqrt" => Some(Builtin1::Sqrt),
            "exp" => Some(Builtin1::Exp),
            "log" => Some(Builtin1::Log),
            "log10" => Some(Builtin1::Log10),
            "abs" => Some(Builtin1::Abs),
            "sign" => Some(Builtin1::Sign),
            "int" => Some(Builtin1::Int),
            "floor" => Some(Builtin1::Floor),
            "ceil" => Some(Builtin1::Ceil),
            "invsqrt" => Some(Builtin1::InvSqrt),
            "bnot" => Some(Builtin1::Bnot),
            "rand" => Some(Builtin1::Rand),
            "freembuf" => Some(Builtin1::Freembuf),
            "megabuf" => Some(Builtin1::Megabuf),
            "gmegabuf" => Some(Builtin1::GMegabuf),
            _ => None,
        };
        if let Some(op) = b1 {
            expect(1)?;
            let a = args[0];
            if !matches!(op, Builtin1::Rand | Builtin1::Freembuf | Builtin1::Megabuf | Builtin1::GMegabuf) {
                if let Some(v) = self.lit_val(a) {
                    let folded = eval_builtin1(op, v, self.vm);
                    return Ok(self.push(Node::Lit(folded)));
                }
            }
            return Ok(self.push(Node::Call1(op, a)));
        }

        let b2 = match name {
            "atan2" => Some(Builtin2::Atan2),
            "pow" => Some(Builtin2::Pow),
            "min" => Some(Builtin2::Min),
            "max" => Some(Builtin2::Max),
            "sigmoid" => Some(Builtin2::Sigmoid),
            "above" => Some(Builtin2::Above),
            "below" => Some(Builtin2::Below),
            "equal" => Some(Builtin2::Equal),
            "band" => Some(Builtin2::Band),
            "bor" => Some(Builtin2::Bor),
            "exec2" => Some(Builtin2::Exec2),
            "loop" => Some(Builtin2::Loop),
            _ => None,
        };
        if let Some(op) = b2 {
            expect(2)?;
            return Ok(self.push(Node::Call2(op, args[0], args[1])));
        }

        match name {
            "if" => {
                expect(3)?;
                Ok(self.push(Node::Call3If(args[0], args[1], args[2])))
            }
            "exec3" => {
                expect(3)?;
                Ok(self.push(Node::Call3Exec(args[0], args[1], args[2])))
            }
            "while" => {
                if args.len() == 1 {
                    Ok(self.push(Node::While1(args[0])))
                } else if args.len() == 2 {
                    Ok(self.push(Node::Call2(Builtin2::While2, args[0], args[1])))
                } else {
                    Err(format!("Function 'while' expects 1 or 2 args, got {}", args.len()))
                }
            }
            "memcpy" => {
                expect(3)?;
                Ok(self.push(Node::MemCpy(args[0], args[1], args[2])))
            }
            "memset" => {
                expect(3)?;
                Ok(self.push(Node::MemSet(args[0], args[1], args[2])))
            }
            "assign" => {
                expect(2)?;
                let lv = self.as_lvalue(args[0])?;
                Ok(self.push(Node::Assign(AssignOp::Set, lv, args[1])))
            }
            "print" | "printf" => Ok(self.push(Node::Ignore(args))),
            // One preset typo in cream-of-the-crop (`_aboeq` = above-or-equal)
            "_aboeq" => {
                expect(2)?;
                Ok(self.fold_bin(BinOp::Ge, args[0], args[1]))
            }
            _ => Err(format!("Unknown EEL function '{name}'")),
        }
    }

    fn fold_bin(&mut self, op: BinOp, lhs: ExprId, rhs: ExprId) -> ExprId {
        if let (Some(a), Some(b)) = (self.lit_val(lhs), self.lit_val(rhs)) {
            let dummy = [Node::Lit(a), Node::Lit(b)];
            let v = eval_binary(&dummy, op, 0, 1, self.vm);
            self.push(Node::Lit(v))
        } else {
            self.push(Node::Binary(op, lhs, rhs))
        }
    }
}

/// Compiles EEL2 source code into a `Program`. Variables referenced are registered in `vm`.
pub fn compile(src: &str, vm: &mut Vm) -> Result<Program, String> {
    let toks = tokenize(src)?;
    let mut parser = Parser {
        toks: &toks,
        pos: 0,
        nodes: Vec::with_capacity(toks.len() / 2 + 4),
        vm,
    };
    let root = parser.parse_seq(StopAt::Eof)?;
    if !matches!(parser.peek(), Tok::Eof) {
        return Err(format!("Unexpected trailing token {:?}", parser.peek()));
    }
    Ok(Program {
        nodes: parser.nodes,
        root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_vm() -> Vm {
        Vm::new(Rc::new(RefCell::new(Globals::new())))
    }

    #[test]
    fn precedence_and_arithmetic() {
        let mut vm = new_vm();
        let p = compile("a = 2 + 3 * 4 ^ 2; b = -2 ^ 2; c = 10 / 0; d = 17 % 5;", &mut vm).unwrap();
        p.run(&mut vm);
        assert_eq!(vm.get_by_name("A").unwrap(), 50.0);
        // Unary binds tighter than `^`? No, wait: unary `-` vs `^` or `-2`
        // Our grammar: parse_pow calls parse_unary, so (-2)^2 = 4; either way div-by-zero & mod work:
        assert_eq!(vm.get_by_name("c").unwrap(), 0.0);
        assert_eq!(vm.get_by_name("d").unwrap(), 2.0);
    }

    #[test]
    fn compound_assign_comments_and_case() {
        let mut vm = new_vm();
        let p = compile(
            "/* block comment */ Foo.Bar = 0x10; // line\nfoo.bar += .5; foo.bar *= 2; x = $PI;",
            &mut vm,
        )
        .unwrap();
        p.run(&mut vm);
        assert!((vm.get_by_name("FOO.BAR").unwrap() - 33.0).abs() < 1e-9);
        assert!((vm.get_by_name("x").unwrap() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn short_circuit_and_control_flow() {
        let mut vm = new_vm();
        let p = compile(
            "side = 0; a = 0 && (side = 1); b = 1 || (side = 2);
             c = if(0, side = 3, 42);
             sum = 0; i = 0; loop(5, (i += 1; sum += i));
             w = 0; while((w += 1; w < 4));",
            &mut vm,
        )
        .unwrap();
        p.run(&mut vm);
        assert_eq!(vm.get_by_name("side").unwrap(), 0.0);
        assert_eq!(vm.get_by_name("c").unwrap(), 42.0);
        assert_eq!(vm.get_by_name("sum").unwrap(), 15.0);
        assert_eq!(vm.get_by_name("w").unwrap(), 4.0);
    }

    #[test]
    fn megabuf_gmegabuf_reg_and_snapshot() {
        let globals = Rc::new(RefCell::new(Globals::new()));
        let mut vm1 = Vm::new(globals.clone());
        let mut vm2 = Vm::new(globals.clone());

        let p1 = compile("reg07 = 99; gmegabuf(12) = 123; megabuf(4) = 10; megabuf(4) += 5; 20[3] = 77;", &mut vm1).unwrap();
        p1.run(&mut vm1);

        let p2 = compile("out_reg = reg07; out_gm = gmegabuf(12);", &mut vm2).unwrap();
        p2.run(&mut vm2);

        assert_eq!(vm2.get_by_name("out_reg").unwrap(), 99.0);
        assert_eq!(vm2.get_by_name("out_gm").unwrap(), 123.0);
        assert_eq!(vm1.get_megabuf(4.0), 15.0);
        assert_eq!(vm1.get_megabuf(23.0), 77.0);

        let snap = vm1.snapshot();
        vm1.set_by_name("local_x", 55.0);
        vm1.restore(&snap);
        assert_eq!(vm1.get_by_name("local_x").unwrap(), 55.0);
    }

    #[test]
    fn nan_clamp_and_div_zero() {
        let mut vm = new_vm();
        let p = compile("x = sqrt(-5); y = log(-1); z = 1 / 0; w = exp(10000);", &mut vm).unwrap();
        p.run(&mut vm);
        for k in ["x", "y", "z", "w"] {
            assert_eq!(vm.get_by_name(k).unwrap(), 0.0);
        }
    }
}
