---
html_theme.sidebar_secondary.remove: true
html_theme.sidebar_primary.remove: true
---

# 🎓 Things I learned

This section is dedicated to things I learned during this internship. It then list the main errors I made (at least the ones I'm aware of).

## 1 -  Mathematical formulae are a good start

Before this intership I was still thinking with a very logician mindset, i.e. considering all the differents minimal steps to go from a problem to a solution. This mindset was conveniant until here because I never made such complexe physical simulations (or at least, without already prepared physical parts). With the radiative transfer one however, I realized how powerfull it is to solve the problem mathematically before trying to code it... which seems evident telling it now, but as I come from a logician background, I had to change of mindset which is not that evident.

## 2 - Unit tests are more than necessary

I spent a considerable amount of time rechecking and rewriting code where I suspected a bug. This problem is directly related to the fact that I did the program in an imperative and not modular form. Write a modular code is usually an habitude but I was affraid to spend too much time on it... I was wrong. Such complexe codes needs to be modular and have unit tests to avoid rechecking/rewriting code. Unfortunately, at the time I write these lines, the physical part is now over so I will not spend time writing these unitary tests (I'm not even yet familiar with unit test libs). But I will definitly think about it from the begining of my future projects.

## 3 - Astropy.units is awsome

I started to use astropy.units to handle the units of the data. This is a very powerful tool that create a lot of errors when writing the code for the first time, but enforce to have a much more robust physical python code (by sacrifying a bit of performances, but critical parts can usually be computed without the dimensions of the quantities). Maybe it's a bit overkill if I write unit tests, but at least it's very usefull for quick physical simulation prototyping.

## 4 - The importance of breaks

By wanting to be efficient, I sometimes tried to stay focus on a problem until it was solved. I spent entire days doing so... but in the end I quickly solved these problems the next morning. 
In purely informatic projects, the problems are almost always algorithmic and some very simple routines are enough to solve them without having a 100% active brain. With physical problems however, "bugs" often require to think a lot more about the fundamental physics behind the problem.

## (Preliminary) Conclusion

Most of these problems have a common part: my desire to go fast, straight to the point. It's a good quality in most of my off-studies activities, but I will need to work on that to be more efficient in a sciencitific context.