#[test]
fn generation_types_are_distinct() {
    // Fixtures include the source because these types are intentionally crate-private.
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/generation_legal.rs");
    t.compile_fail("tests/ui/generation_assignment_mismatch.rs");
    t.compile_fail("tests/ui/generation_comparison_mismatch.rs");
}
