# Pull Request

## Summary
- What does this change do?
- Why is it needed?
- Context/links (issue, docs):

Closes: #
Related: #

## Type of change
- [ ] Feature
- [ ] Bug fix
- [ ] Refactor/cleanup
- [ ] CI/Chore
- [ ] Documentation

## How has this been tested?
- [ ] Local build with CMake/Ninja succeeds
- [ ] Unit/integration tests added or updated
- [ ] Manual test notes:

## Pre-submit checklist
- [ ] CI is green (jobs: quality, sanitizers)
- [ ] clang-format applied (or run: `clang-format -i`)
- [ ] clang-tidy warnings addressed or justified with NOLINT and comments
- [ ] cppcheck clean or suppressions added with rationale
- [ ] cpplint clean
- [ ] Include-What-You-Use (advice) reviewed, unnecessary headers removed
- [ ] AddressSanitizer + UBSanitizer run clean (locally or via CI)
- [ ] No new compiler warnings (we treat warnings as errors)
- [ ] Tests cover new/changed code; ctest passes
- [ ] Docs updated (README, comments, public headers)
- [ ] Breaking changes documented in PR and commit message
- [ ] Secrets/credentials not included; security impact considered

## Screenshots / Logs (if UI or relevant)

## Notes for reviewers
- Areas to focus on:
- Follow-up work:
