- In IROL_SKY branch
    - Add stage
    - Commit with your message
    - Push to remote branch

```bash
git add .
git commit -m "<YOUR-COMMENT>"
git push origin IROL_SKY
```

- If your current branch is not IROL_SKY
```bash
git branch -a
(FIND YOUR BRANCH)
git checkout <YOUR-BRANCH>
```

- Update remote branch & update master branch (sync to remote)
```bash
git remote update
git checkout master
git pull origin master
```

```bash
git merge <YOUR-BRANCH>
```

> Conflict Fixing

- Push to master branch
```bash
git add .
git commit -m "<YOUR-COMMENT>"
git push origin master
```