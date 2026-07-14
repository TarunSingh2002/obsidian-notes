### IAM Basics
- IAM = Identity and access management,
- Global service
- Users are people within your organization and can be grouped
- Groups only contains users and not other groups
- 1 user can belong to multiple group or no group
- For example 
	- ![[Pasted image 20260627175002.png]]
	- 3 Groups
		- Group 1
			- Alice
			- Bob
			- Charles
		- Group 2
			- David
			- Edward
		- Group 3
			- Charlies
			- David
	- Here Fred does not belong to any group 
	- Here Charles and David belong to more then one group 
- Users and Groups can be assigned json document called **policies** and these policies define the permissions of the users
	- Example of Policy ![[Pasted image 20260627175752.png]]
- Least privilege principle = Don't give more permissions then a user need 

### Root user

![[Pasted image 20260627180111.png]]

The account id here means => Root user

### How to create a user
- IAM console -> IAM users -> create user -> Give user name (like Test-User-1) -> check 'Provide user access to the AWS Management Console - optional' -> custom password -> next -> 'Get started with groups - create group' -> give group name and policies access -> next -> create user
-  how a IAM user looks like ![[Pasted image 20260709004216.png]]
### How to create account Alias 
- IAM -> Dashboard -> account Alias create
- It change the signin url from 
	- https://546804384519.signin.aws.amazon.com/console
	- https://aws-tarun-2002.signin.aws.amazon.com/console
### Running 2 different account on a same browser 
- Like for example - 1 root user and 1 iam user
- with the help of "Multi session support"
- ![[Pasted image 20260709005054.png]]
### IAM Policies Inheritance
