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
* Policy attached to a group → applies to ALL users in that group
* User in multiple groups → inherits policies from ALL of them
	* Example: Charles is in Developers + Audit team → he gets both policies
	* ![[Pasted image 20260717010755.png]]
* Inline policy = policy attached directly to 1 user only (not through a group)
	* That user can be in a group or in no group, doesn't matter
* **Conflict rule → explicit Deny ALWAYS beats Allow (exam important)**
	* For Example Charles is in Developers + Audit group → he gets both policies
	* If Developer group Deny something which is allowed by Audit group -> it will be still deny
### ARN (Amazon Resource Name)
* ARN = unique address of ANY thing in AWS (like full home address)
* Format → arn:partition:service:region:account-id:resource
* Example → arn:aws:iam::123456789012:root
	* aws → partition (almost always just "aws")
	* iam → service name
	* :: → region is empty, because IAM is global
	* 123456789012 → my 12 digit account id
	* root → means the whole account
* Find any ARN → click the user/role/resource in console → ARN shown on its page
* Find account id → top right corner of console
### IAM Policy Structure
* Policies = JSON documents
* Top level parts:
	* Version → policy language version, almost always "2012-10-17"
	* Id → name of the policy (optional)
	* Statement → the main part, can be 1 or many
* Each Statement has:
	* Sid → statement id (optional)
	* Effect → Allow or Deny
	* Principal → WHO it applies to (account / user / role)
	* Action → WHAT API calls (like s3:GetObject)
	* Resource → ON WHAT the action applies (like a bucket)
	* Condition → WHEN it applies (optional)
* Trick to read any policy like a sentence:
	* Effect + Principal + Action + Resource = "Allow this user to do these API calls on this resource"
```JSON
{
  "Version": "2012-10-17",
  "Id": "S3-Account-Permissions",
  "Statement": [
    {
      "Sid": "1",
      "Effect": "Allow/Deny",
      "Principal": { "AWS": "arn:aws:iam::123456789012:root" },
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::my-bucket/*"]
    }
  ]
}
```
#### Where do Principal / Action / Resource values come from
* Principal → WHO. Value = an ARN
	* whole account → arn:aws:iam::ACCOUNT-ID:root
	* one user → arn:aws:iam::ACCOUNT-ID:user/Bob
	* one role → arn:aws:iam::ACCOUNT-ID:role/MyRole
	* AWS service → { "Service": "ec2.amazonaws.com" }
	* "AWS" key just means "this principal is an AWS account/user/role"
	* IMPORTANT → policy attached to user/group = NO Principal needed, AWS already knows who
* Action → WHAT. Value = service:APIName
	* fixed list made by AWS, never made up by us
	* Object = file in S3 language → s3:GetObject = download file, s3:PutObject = upload file
	* find the list → visual editor in IAM console (checkboxes) or search "S3 actions Service Authorization Reference"
	* wildcards → s3:* = all s3 actions, s3:Get* = all read type actions
* Resource → ON WHAT. Value = ARN of the target
	* arn:aws:s3:::my-bucket → the bucket, arn:aws:s3:::my-bucket/* → files inside it
	* "*" alone = everything
* Real life → nobody types policies from blank page. Use visual editor, AWS managed (pre-made) policies, or copy from docs
### Adding Permission to a user Directly
Got to 'IAM users' -> Select the user -> Permissions -> Add permissions -> Attach Policy directly -> search 'IAMRead' -> next -> Add Permissions
### We can also create a custom policy 
Go to 'IAM Policy' -> Create Policy -> use either custom visual editor or Json editor.
### IAM - Password Policy
IAM -> Account setting -> Password Policy -> Edit -> you can either use IAM default or Custom
* Password policy ≠ JSON permission policy → it's just an ACCOUNT SETTING
* Only 1 password policy per AWS account → nothing to attach
* Auto applies to ALL IAM users → existing + future ones
* Does NOT apply to root (root is outside IAM)
* Can't apply to only some users → all or nothing
* Existing user → old password works until he changes it, then new rules apply
	* Force now → password expiration OR "require password reset at next sign-in"
### 3 Ways to Access AWS
* Management Console → web interface → protected by password + MFA
* CLI → commands from your terminal → protected by Access Keys
* SDK → code inside your application → protected by same Access Keys
### Access Keys
* Generated from the Management Console
* Each user creates + manages their OWN keys
* Treat them like login credentials, NEVER share
	* Access Key ID = like username
	* Secret Access Key = like password
* Secret is downloadable only ONCE → at creation time
#### how to create Access Keys
- Users -> Security credentials -> Access Keys ->Create Access key -> CLI -> next -> create access key -> Done

### Set up AWS CLI
- aws configure -> run this command
- aws iam list-users -> run this command to check list of user 
### CloudShell
- Browser Terminal to access the AWS, inside aws console
- ![[Pasted image 20260720024829.png]]This icon
- Files persist between sessions 
	- e.g. `echo "test" > demo.txt` → restart CloudShell → file is still there 
	- Storage = 1 GB per region, free, only inside home directory ($HOME) 
	- Files outside $HOME get wiped when session ends 
	- If you don't use CloudShell in a region for 120 days → that region's data gets auto-deleted

### IAM Roles
- Problem → some AWS services need to perform actions on YOUR account, on your behalf 
	- To do any action, they need permissions → exactly like users do
- IAM Role = just like an IAM user, BUT not for physical people → meant to be used by AWS services
- Example → EC2 instance (virtual server) wants to access some info from AWS:
	- Create an IAM Role + attach it to the EC2 instance → together they act as one entity 
	- EC2 makes a call → it uses the role → if the role's permissions allow it → call succeeds
IAM -> Roles -> create role -> AWs service -> USE Case ('EC2') -> Ec2 -> Next -> add any policy like 'IAMReadOnlyAccess' -> NEXT -> Add role name and description  -> create role

### IAM Security tools
- Two tools to audit security in IAM: 
- IAM Credentials Report → ACCOUNT-level 
	- Report containing ALL users in your account + status of their various credentials 
	- IAM -> Access Reports -> Credential reports -> Download credential reports 
- IAM Access Advisor → USER-level 
	- Shows service permissions GRANTED to a user + WHEN those services were last accessed
	- Purpose → spot permissions that are never used → remove them → principle of least privilege
	- IAM -> IAM Users -> Select a user -> Last Accessed

### IAM Guidelines & Best Practices 
- Don't use root account → only for initial AWS account setup, nothing else 
- 1 AWS user = 1 physical user → friend wants to use AWS? → create a NEW user, never share yours 
- Assign users to groups → manage permissions at the GROUP level 
- Create a strong password policy 
- Use + enforce MFA 
- Create and use ROLES whenever giving permissions to AWS services (incl. EC2 instances) 
- Using CLI / SDK? → generate access keys 
- Access keys = just like passwords → very secret, keep them to yourself 
- Audit permissions → Credentials Report + Access Advisor 
- NEVER ever share IAM users & access keys