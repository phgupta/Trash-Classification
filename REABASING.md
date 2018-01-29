# We start out with making a new branch for a feature
When making a branch we can follow a standard convention like the following:

| Purpose | Branch name |
| ------- | ----------- |
| Fix | fix/*your-short-branch-name* |
| New Feature | feature/*your-short-branch-name* |
| Documentation changes | docs/*your-short-branch-name* |

After making the branch we should push the branch to origin.
```sh
$ git push --set-upstream origin branch-name-goes-here
```
After this, it will show up on the repo and you can push using `git push`

# Pushing and merging your feature
We can follow a simple and general procedure so we can avoid dealing to merge conflicts. It is a good idea to always fetch from the repo (not pull), this will retrieve 
changes from origin but not physically apply them. `git pull` can be thought of as a `git fetch` and `git merge`

```sh
$ git fetch
```
# Getting ready to merge your changes to master
You would want to put your changes on top of master first before merging. Doing this keeps a linear history of our progress. The following command may have merge conflicts.
```sh
$ git rebase --interactive origin/master
```
The interactive option allows us to pick and choose commits that we want to place on top of master. It opens up text editor that will prompt you to choose the commits. You should `squash` the trivial commits by editing the file and writing squash in front of the commit. If you are going to Merge Request, please try to squash commits down to a few (1-3), but prefereably one with a message briefly explaining what you've done.

For example my rebase for this resource file was:
```
pick 36642de Created resources file
pick 6a2c4ba Added some text to file
pick b076927 Small fixup
pick efdc0b4 Changing the markdown

# Rebase ef89529..efdc0b4 onto ef89529 (4 commands)
#
# Commands:
# p, pick = use commit
# r, reword = use commit, but edit the commit message
# e, edit = use commit, but stop for amending
# s, squash = use commit, but meld into previous commit
# f, fixup = like "squash", but discard this commit's log message
# x, exec = run command (the rest of the line) using shell
# d, drop = remove commit
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out
```
Clearly I still have some minor commits that I can combine. I change my commits to squash and I rename them.
```
r 36642de Added a resources file
f 6a2c4ba Added some text to file
f b076927 Small fixup
f efdc0b4 Changing the markdown
# Rebase ef89529..efdc0b4 onto ef89529 (4 commands)
#
# Commands:
# p, pick = use commit
# r, reword = use commit, but edit the commit message
# e, edit = use commit, but stop for amending
# s, squash = use commit, but meld into previous commit
# f, fixup = like "squash", but discard this commit's log message
# x, exec = run command (the rest of the line) using shell
# d, drop = remove commit
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out
```
Now it will show up as one large commit instead of many small ones.

After you've successfully rebased, you can push to your branch by running
```sh
$ git push --force
```

