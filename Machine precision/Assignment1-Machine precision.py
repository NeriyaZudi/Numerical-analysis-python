"""
 *   Authors: Maor Arnon (ID: 205974553) and Nehorai (ID:XXXXXXXXXX)
 *            Emails: maorar1@ac.sce.ac.il    XXXXXX@Gmail.com
 *   Department of Computer Engineering - Assignment 1 - Numeric Analytics
"""

# Question 1

print("\nThis is the calculation before I fixed the inaccuracy issue.")
print((abs(3.0 * (4.0 / 3.0 - 1) - 1)))

print("\nBecause of this part of the calculation, When the cpu tries to save the value 1/3 it can't save precise value "
      "dou to binary value limitations.")
print("4/3 -1= " + str(4 / 3 - 1))

print("\nSo after we multiply this imprecise value by 3 we get a number close to 1 instead of 1.")
print("3.0*(4.0/3.0-1)= " + str(3.0 * (4.0 / 3.0 - 1)))

print("\nSo when we try to subtract 1 from it will become a very small negative number when it should give us 0.")
print("(3.0*(4.0/3.0-1)-1)= " + str((3.0 * (4.0 / 3.0 - 1) - 1)))

print("\nWe can fix that by using the round function like so:")
print("abs(round(3.0*(4.0/3.0-1)-1)) = " + str(abs(round(3.0 * (4.0 / 3.0 - 1) - 1))))

# Question 2

print(
    """
Machine precision is the smallest number eps such that the difference between 1 and 1 + eps is nonzero, ie.
, it is the smallest difference between two numbers that the computer recognizes. 
On a 32 bit computer, single precision is 2-23 (approximately 10-7) while double precision is 2-52 (approximately10-16).

I got this definition from a simple google search
From this site : 'https://praveen.tifrbng.res.in/computing/precision'
"""
)

# Question 3

print(
    """
We can find our computer epsilon by dividing a number by 2 in a while loop
until we 0 the previous returned value from the division is the smallest number our machine can evaluate.
""")

"""
number - The starting number to be divided 1 for example (can be any positive number).
iteration - Only to visualize the number of division required to find our machine epsilon.
flag - Shows when we found the machine epsilon.
"""

number = 1
iteration = 0
flag = True
print("(0) ", end="")
while flag:
    iteration += 1
    print(("{0} \n({1}) {0} / 2 = ".format(number, iteration)), end="")
    epsilon = number
    number /= 2
    if number == 0:
        flag = False
        print('  0')

print("\nEpsilon equals to {}".format(epsilon))
