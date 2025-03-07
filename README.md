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

- Come back to YOUR BRANCH
```bash
git checkout <YOUR-BRANCH>
```

- Merge master
```bash
git merge master
git add .
git commit -m "<YOUR-COMMNET>"
git push origin <YOUR-BRANCH>
```

> 그냥 적당히 자주 쓰는 것
```bash
git rm --cached <DIRECTORY> # 특정 디렉토리의 캐시 삭제
git rm -r --cached . # 모든 파일의 캐시 삭제 (최후의 수단)
```

```bash
git reset --soft HEAD~<N> # N개 전의 커밋으로 돌아가기. N이 1이기를 기도하면 됨
```
