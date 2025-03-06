stairs = []
dp = [1]
step_result = [0]

num = int(input())
for i in range(num):
    stairs.append(int(input()))
   
for step in range(1, num-1):
    if dp[step - 1] == num:
        break
    
    if len(dp) >= 2:
        if dp[step-1] - dp[step-2] == 1:
            dp.append(dp[step - 1] + 2)

    elif stairs[num - dp[step - 1]] + stairs[num - (dp[step - 1] + 2)] > stairs[num - dp[step - 1]] + stairs[num - (dp[step - 1] + 1)]:
        dp.append(dp[step - 1] + 2)
    
    elif stairs[num - dp[step - 1]] + stairs[num - (dp[step - 1] + 2)] < stairs[num - dp[step - 1]] + stairs[num - (dp[step - 1] + 1)] and dp[step-1] - dp[step-2] != 1:
        dp.append(dp[step - 1] + 1)










