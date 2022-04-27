# def escape(instr):
# 	escapetime=0
# 	nows=0
# 	thises=0
# 	for i in range(len(instr)):
# 		if instr[i]=="#":
# 			nows=not nows
# 			if nows==0:
# 				escapetime+=thises
# 				thises=0
# 		else:
# 			if nows==1:
# 				if instr[i-1]=="!" and ord(instr[i])>=ord("a") and ord(instr[i])<=ord("z"):
# 					thises+=1
# 	return escapetime

# def escape(s):
#     l = len(s)
#     res = 0
#     h=0
#     sres=0
#     for i in range(l):
        
#         if(s[i]=='#'):
#             if(h==0):
#                 h=1
#                 sres=0
#             elif(h==1):
#                 h=0
#                 res += sres
#         if(h==1 and s[i]=='!' and s[i+1]>'a' and s[i+1]<'z'):
#             sres += 1
#     return res
        

# print(escape("##!r#po#"))
# print(escape("#ab!c#de!f"))
# print(escape("a!de#dwx!re!e##!##sdc!a!f"))

# String = ('G', 'e', 'e', 'k', 's', 'F', 'o', 'r')
  
# Fset1 = frozenset(String)
# print("The FrozenSet is: ")
# print(Fset1)
  
# # To print Empty Frozen Set
# # No parameter is passed
# print("\nEmpty FrozenSet: ")
# print(frozenset())

def findoptimal(N):
 
    # The optimal string length is
    # N when N is smaller than 7
    if (N <= 6):
        return N
 
    # An array to store result of
    # subproblems
    screen = [0]*N
 
    # Initializing the optimal lengths
    # array for until 6 input
    # strokes.
     
    for n in range(1, 7):
        screen[n-1] = n
 
    # Solve all subproblems in bottom manner
    for n in range(7, N + 1):
     
        # Initialize length of optimal
        # string for n keystrokes
        screen[n-1] = 0
 
        # For any keystroke n, we need to
        # loop from n-3 keystrokes
        # back to 1 keystroke to find a breakpoint
        # 'b' after which we
        # will have ctrl-a, ctrl-c and then only
        # ctrl-v all the way.
        for b in range(n-3, 0, -1):
         
            # if the breakpoint is at b'th keystroke then
            # the optimal string would have length
            # (n-b-1)*screen[b-1];
            curr = (n-b-1)*screen[b-1]
            if (curr > screen[n-1]):
                screen[n-1] = curr
         
    return screen[N-1]
 
# Driver program
# if __name__ == "__main__":
 
#     # for the rest of the array we
#     # will reply on the previous
#     # entries to compute new ones
#     for N in range(1, 21):
#         print("Maximum Number of A's with ", N, " keystrokes is ",
#                 findoptimal(N))
def asdf(n):
    res=[]
    for i in range(n):
        if i<3:
            res.append(i)
        else:
            res.append(max(res[i-1]+1,res[i-3]*2))
    return res
# print(asdf(12))

def check(stalls, cows, distance):
    stalls = sorted(stalls)
    last = stalls[0]
    cows -= 1
    l = len(stalls)
    for i in range(1,l):
        if(stalls[i]-last >= distance):
            print(cows,last)
            cows -= 1
            last = stalls[i]
            if(cows == 0):
                return True
    return False

print(check([1,2,4,8,9],3,4))
