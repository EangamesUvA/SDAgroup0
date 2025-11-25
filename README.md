# Running the project
Run file `python filename.py` or `python3 filename.py` and if necessary it will automatically install all required pip packages (using `requirements.txt` and `requirements.py`).

## Testing
To run all tests `pytest` or to run a specific test `pytest [filename]`.

# Git help instructions
## Git crash course
For more info about git and some background information look at this: [Git crash course](https://gist.github.com/brandon1024/14b5f9fcfd982658d01811ee3045ff1e)

## Git installation
Go to [Git Install](https://git-scm.com/install/) and make sure you have git installed.

If you haven't already use:
- `git config --global user.name "Your Name"`
- `git config --global user.email "your.email@example.com"`
to setup github account.

## Repository installation
Make your own folder on your computer, go to it in the terminal, and then:
- `git clone https://github.com/EangamesUvA/SDAgroup0.git` if using the steps above or
- `git clone git@github.com:EangamesUvA/SDAgroup0.git` if using ssh.

## Code editor
When using a code editor like vs code, you don't really need to know all below, still please read this to make sure that if a problem arises you can solve it yourself.

## First pull/push
For the first time pulling or pushing the current branch of the git repository use:
- `git push -u origin <branch_name>`
- `git pull origin <branch_name>`

## Making your own branch
```bash
git checkout main
git pull
git checkout -b <new_branch_name>
```
If you want to copy all files from another branch exchange *main* for the branch name.

## Switching branch
Use `git checkout <branch_name>` to go to another repository but first make sure that you don't have any uncommited changes (by trying to make a commit).

## Making a commit
```bash
git add .
git add <file_name> # for new filenames
git commit -m "commit message"
git push
```
Try to come up with a good commit message but small changes *can* use something like ".", also try to commit once in a while and not commit the whole feature at once.

## Which branch to use
When adding stuff to code or working on one document with multiple people you should use your own branch (you could work together on the same branch but conflicts will be harder for code).
When done with your section of code or feature, you should open up a pull request on the github website or message me that the branch is done, then I will merge the branch and you can go on with your work.

**Never** commit and push to the main branch without explicit permission or knowledge of what might happen as merge conflicts on the main branch are not fun to work out (I did this before).

## Branch information
Use `git branch` to show all active branches and `git branch -a` to show all active and remote branches. This also shows the current branch with a '*' before it.

# Python
## Style guide
For the style guide you can look at this: [Style guide](https://peps.python.org/pep-0008/),
but the main things to worry about:
1. Try to keep lines short (when possible no more than 80 characters wide).
2. Try to remove unnecessary indentation by factoring out functions (I would do no more than 3 to 5 indentations).
3. Keep function names informative but not too long (as that will conflict with point 1 easier).
4. Make good variable names (no short names of only letters or two letters, exception is if a mathematical formula gives a name to that variable, use that instead)
5. Use *snake_case* for function and variable names, *UPPER_CASE_SNAKE_CASE* for global variables, and *PascalCase* for class names (not the objects generated from it as those are variables).
