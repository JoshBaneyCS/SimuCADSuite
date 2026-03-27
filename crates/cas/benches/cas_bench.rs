use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simucad_cas::derivative::differentiate;
use simucad_cas::evaluator::{evaluate, Environment};
use simucad_cas::parser::parse;
use simucad_cas::simplify::simplify;

const EXPR: &str = "x^3 + 2*x^2 + sin(x)";

/// A deeply nested expression to stress-test simplification.
fn deep_nested_expr() -> String {
    // Build: ((((x + 1) * x + 1) * x + 1) ... ) — 50 layers deep
    let mut expr = String::from("x");
    for _ in 0..50 {
        expr = format!("({expr} + 1) * x");
    }
    expr
}

fn bench_parse(c: &mut Criterion) {
    c.bench_function("cas_parse", |b| {
        b.iter(|| {
            parse(black_box(EXPR)).unwrap();
        });
    });
}

fn bench_evaluate(c: &mut Criterion) {
    let ast = parse(EXPR).unwrap();
    let mut env = Environment::new();
    env.set("x", 2.0);

    c.bench_function("cas_evaluate", |b| {
        b.iter(|| {
            evaluate(black_box(&ast), black_box(&env)).unwrap();
        });
    });
}

fn bench_differentiate(c: &mut Criterion) {
    let ast = parse(EXPR).unwrap();

    c.bench_function("cas_differentiate", |b| {
        b.iter(|| {
            differentiate(black_box(&ast), "x").unwrap();
        });
    });
}

fn bench_simplify(c: &mut Criterion) {
    let ast = parse(EXPR).unwrap();
    let derived = differentiate(&ast, "x").unwrap();

    c.bench_function("cas_simplify", |b| {
        b.iter(|| {
            simplify(black_box(&derived));
        });
    });
}

fn bench_simplify_deep(c: &mut Criterion) {
    let deep = deep_nested_expr();
    let ast = parse(&deep).unwrap();

    let mut group = c.benchmark_group("cas_simplify_deep");
    group.sample_size(10);
    group.bench_function("50_layers", |b| {
        b.iter(|| {
            simplify(black_box(&ast));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_parse,
    bench_evaluate,
    bench_differentiate,
    bench_simplify,
    bench_simplify_deep,
);
criterion_main!(benches);
