#[test]
fn generation_types_are_distinct() {
    // Fixtures include the source because these types are intentionally crate-private.
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/generation_legal.rs");
    t.compile_fail("tests/ui/generation_assignment_mismatch.rs");
    t.compile_fail("tests/ui/generation_comparison_mismatch.rs");
    t.pass("tests/ui/public_api_pass.rs");
    t.compile_fail("tests/ui/cannot_construct_work.rs");
    t.compile_fail("tests/ui/no_generation_accessor.rs");
    t.compile_fail("tests/ui/cannot_alternate_session_target.rs");
}
