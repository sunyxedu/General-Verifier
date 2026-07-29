-- Verification target:
-- claim: 对所有 1<=n<=300, 60 整除 n^5-n
-- obligation: 证明必须给出 n^5-n 的正确代数分解, 例如
--   n^5 - n = n*(n-1)*(n+1)*(n^2+1), 且分解式在后续整除性论证中被无误地使用.
-- bound method:
--   (a) 用 `decide` 逐点检验分解式数值相等 (确证分解式本身无代数错误)
--   (b) 把有界全称命题 60|n^5-n (1<=n<=300) 编码为可计算 Bool, 用
--       #eval 和 native_decide 做穷举检查

-- (a) 分解式 n^5-n = n*(n-1)*(n+1)*(n^2+1) 的逐点数值检验
example : (1:Nat)^5 - 1 = 1*(1-1)*(1+1)*(1^2+1) := by decide
example : (2:Nat)^5 - 2 = 2*(2-1)*(2+1)*(2^2+1) := by decide
example : (3:Nat)^5 - 3 = 3*(3-1)*(3+1)*(3^2+1) := by decide
example : (7:Nat)^5 - 7 = 7*(7-1)*(7+1)*(7^2+1) := by decide
example : (10:Nat)^5 - 10 = 10*(10-1)*(10+1)*(10^2+1) := by decide
example : (300:Nat)^5 - 300 = 300*(300-1)*(300+1)*(300^2+1) := by decide

-- (a') 分解式对 [1,300] 内所有 n 的穷举核对, native_decide 确证
def factCheck : Bool := (List.range 300).all (fun i => let n := i + 1;
  n^5 - n == n*(n-1)*(n+1)*(n^2+1))

#eval factCheck
example : factCheck = true := by native_decide

-- (b) 有界全称命题 '∀ 1<=n<=300, 60 | n^5-n' 的编码与穷举检查
def check : Bool := (List.range 300).all (fun i => let n := i + 1;
  (n^5 - n) % 60 == 0)

#eval check

-- 若 check = false, 定位第一个反例 (0-based index, n = i+1)
def firstFail : Option Nat := (List.range 300).find? (fun i => let n := i + 1;
  (n^5 - n) % 60 != 0)

#eval firstFail

-- 反例处的具体数值 (n = 2)
#eval (2:Nat)^5 - 2
#eval ((2:Nat)^5 - 2) % 60

-- native_decide 内核/编译器可信的反例确证: check = false
example : check = false := by native_decide
