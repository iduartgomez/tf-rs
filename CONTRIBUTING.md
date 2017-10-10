# Contributing to tf-rs

If you want to contribute any contribution is helpful and welcome! Look at issues, and if there aren't any that you can contribute to feel free to open issues for feature requests, discussion or fixes.

Additions and improvements to documentation, examples, guides, etc. are welcome.

## Pull requests

#### Commit Message Format

Try to stick to this format for clarity (inspired by AngularJS project). Each commit message consists of at least a **header**, and a optionally a **body**. The header has a special format that includes a **type** and a **subject**:

```
<type>: <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

#### Type
Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **docs**: Documentation only changes
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **refactor**: A code change that neither fixes a bug or adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing tests
* **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation

#### Subject

The subject contains succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize first letter
* no dot (.) at the end

#### Body

Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes"
The body should include the motivation for the change and contrast this with previous behavior.

#### Footer

The footer should contain any information about **Breaking Changes** and is also the place to
reference GitHub issues that this commit **Closes**.

The last line of commits introducing breaking changes should be in the form `BREAKING CHANGE: <desc>`
