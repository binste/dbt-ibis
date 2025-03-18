# Advanced
## Use `dbt` command instead of `dbt-ibis`
If you want to continue to use `dbt` instead of `dbt-ibis` on the command line, you can configure an alias in your shell. If you use bash, you can add the following to your `~/.bashrc` file:
```
alias dbt="dbt-ibis"
```

See [here for more detailed instructions if you use Bash](https://linuxize.com/post/how-to-create-bash-aliases/) and [here for Zsh](https://linuxhint.com/configure-use-aliases-zsh/).

## CI/CD integration
As `dbt-ibis` compiles your `.ibis` files into `.sql`, it can be useful to check in your CI/CD pipeline if these files are in sync, i.e. a run of `dbt-ibis precompile` should not change any `.sql` files anymore. You can achieve this with something like:

```bash
#!/bin/bash

dbt-ibis precompile
# This gets the paths of all files which were either deleted, modified
# or are not yet tracked by Git
files=`git ls-files --deleted --modified --others --exclude-standard`
# Depending on the shell it can happen that 'files' contains empty
# lines which are filtered out in the for loop below
files_cleaned=()
for i in "${files[@]}"; do
# Skip empty items
if [ -z "$i" ]; then
    continue
fi
# Add the rest of the elements to a new array
files_cleaned+=("${i}")
done
if [ ${#files_cleaned[@]} -gt 0 ]; then
    echo "The dbt-ibis precompile command modified the following files:"
    echo $files
    exit 1
fi
```

## Potential closer integration with DBT
There are [discussions](https://github.com/dbt-labs/dbt-core/pull/5274#issuecomment-1132772028) on [adding a plugin system to dbt](https://github.com/dbt-labs/dbt-core/issues/6184) which could be used to provide first-class support for other modeling languages such as Ibis (see [this PoC](https://github.com/dbt-labs/dbt-core/pull/6296) by dbt and the [discussion on Ibis as a dataframe API](https://github.com/dbt-labs/dbt-core/discussions/5738)) or PRQL (see [dbt-prql](https://github.com/PRQL/dbt-prql)).

As this feature didn't make it [onto the roadmap of dbt for 2023](https://github.com/dbt-labs/dbt-core/blob/main/docs/roadmap/2023-02-back-to-basics.md), I've decided to create `dbt-ibis` to bridge the time until then. Apart from the limitations mentioned above, I think this approach can scale reasonably well. However, the goal is to migrate to the official plugin system as soon as it's available.
