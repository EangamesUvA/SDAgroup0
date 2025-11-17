# Git help instructions
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
