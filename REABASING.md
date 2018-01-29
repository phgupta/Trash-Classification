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
This will update origin with your commits ontop of the master branch, which will have no merge conflicts.
**NOTE: Please do not push directly onto master yourself, work in your own branches.**

# How to make a merge request
Once you are done with your branch and it's ready to be merged into master, make sure that your changes are rebased ontop of master branch and have no merge conflicts.
Then you can go to our [github repo's branches](https://github.com/phgupta/Trash-Classification/branches), and pressing "New pull request" button. You will be brought to this screen. Here you can type out a name and a description for your merge request. Also **assign someone to review your changes**, and notify them via slack or other way that they have a merge request to look over. So we can all keep track of code quality and what gets merged into master.
# Reviewing a merge Request
If you are chosen to review a merge request do your best to check the code, make sure no merge conflicts exists. 
You can approve a merge request that:
1. Do not have merge conflicts
2. See 
3. The merge request is not behind master. (**NOTE That means other people have updated the master before you and you don't have all the features; in that case you will rebase again and make a new merge request**)
If all looks good, you can press "Rebase and merge".

# How to report an issue
To be organized, we should track all our project issues in one place. Thus, to report an issue go to [issues section on our repo](https://github.com/phgupta/Trash-Classification/issues).
Here you can press "New issue" to create a new issue. Fill out a title for the issue, and in the description be as detailed as possible to recreate the issue and explain what the issue is. I'll add a issue template.
