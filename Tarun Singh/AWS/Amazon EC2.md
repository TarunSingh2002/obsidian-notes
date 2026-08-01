### EC2 Basics
* EC2 = Elastic Compute Cloud = IaaS (Infrastructure as a Service)
* EC2 is not 1 single service, it's a family:
	* EC2 Instances → rent virtual machines
	* EBS Volumes → store data on virtual drives
	* ELB (Elastic Load Balancer) → distribute load across machines
	* ASG (Auto Scaling Group) → scale services up/down
* Core idea of cloud → rent compute on demand, whenever you need
- The public ip = will change when a instance restart that why we need to have elastic ip 
### EC2 Instance Config Options (what YOU choose)
* OS → Linux (most popular), Windows, or macOS
* CPU → how much compute power + cores
* RAM → how much memory
* Storage →
	* Network attached → EBS / EFS
	* Hardware attached → EC2 Instance Store
* Network → card speed, type of public IP
* Firewall rules → Security Group
* Bootstrap script at first launch → EC2 User Data

### EC2 User Data
* Bootstrapping = launching commands when machine starts
* Script runs only ONCE → at FIRST boot, never again
* Purpose → automate boot tasks
	* install updates, install software, download files etc
* More stuff in the script = slower boot time
* Runs as ROOT user → every command has sudo power

### Launching First EC2 Instance
#### Launching
* Steps → EC2 Console -> Instances -> Launch Instances
	* Name → give name tag (like My First Instance)
	* AMI → base image / OS of instance → pick Amazon Linux 2 (free tier eligible), 64-bit x86
	* Instance type → t2.micro (free tier eligible) → types differ by CPU, RAM, cost
	* Key pair → needed to SSH into instance
		* create new → type RSA
		* format → .pem for Mac / Linux / Windows 10+
		* format → .ppk for Windows 7/8 (used with PuTTY)
	* Network settings → security group auto created (launch-wizard-1)
		* Allow SSH traffic from anywhere → port 22
		* Allow HTTP traffic from internet → port 80 (needed for web server)
	* Storage → 8 GB gp2 root volume (free tier = up to 30 GB)
		* "Delete on termination" = Yes by default → terminate instance = volume also deleted
	* Advanced → bottom → User Data → paste script
		* runs ONCE at first launch → installs httpd web server + writes HTML file
* Free tier → 750 hrs/month of t2.micro (= 1 instance running full month)
	* region has no t2.micro → t3.micro is free instead

#### After Launch — Instance Details

* Instance ID → unique identifier
* Public IPv4 → access instance from internet
* Private IPv4 → access instance INSIDE AWS private network
* To see web server → http://PUBLIC-IP
	* MUST use http:// NOT https:// → https gives infinite loading
	* too fast after launch → no page yet, wait few min + refresh

#### Start / Stop / Terminate

* Stop → instance state kept (volume stays), NOT billed for compute
* Terminate → instance deleted permanently
* IMP (exam) → Stop then Start again:
	* Public IPv4 → CAN CHANGE
	* Private IPv4 → NEVER changes

### EC2 Instance Types

* Different instance types → optimized for different use cases
* https://aws.amazon.com/ec2/instance-types/
#### Naming Convention

* Example → m5.2xlarge
	* m → instance CLASS (here = general purpose)
	* 5 → GENERATION (hardware improves → m5 becomes m6 etc)
	* 2xlarge → SIZE within the class (large → 2xlarge → 4xlarge...)
* Bigger size = more CPU + more memory

#### Total 7 types, but The 4 Main Types (exam)

* General Purpose
	* balanced compute + memory + networking
	* use → web servers, code repos, diverse workloads
	* our free tier t2.micro is this type
* Compute Optimized → letter C (C5, C6...)
	* great for high CPU / processor heavy tasks
	* use → batch processing, media transcoding, HPC, ML, gaming servers
* Memory Optimized → letter R (R = RAM), also X1, Z1
	* fast for large datasets in memory (RAM)
	* use → in-memory databases, ElastiCache, real-time big data processing
* Storage Optimized → letters I, D, H1
	* great for lots of local storage access
	* use → OLTP, NoSQL/relational DBs, data warehousing, distributed file systems

### Security Groups (EC2 Firewall)
* Security Group = firewall AROUND your EC2 instance
* Controls traffic allowed IN and OUT of instance
* Contain ONLY allow rules (no deny rules)
* Rules can reference → an IP address range OR another security group
#### Rule Details
* Each rule has → Type, Protocol (TCP), Port, Source (IP range)
* Inbound → traffic from outside INTO instance
* Outbound → traffic from instance OUT to internet
* 0.0.0.0/0 → means everything (all IPs)
* Single IP → just that one computer allowed

#### Defaults
* All INBOUND traffic → BLOCKED by default
* All OUTBOUND traffic → ALLOWED by default

#### Key Facts (exam)
* 1 security group → can attach to MANY instances
* 1 instance → can have MANY security groups
* Locked to Region + VPC → switch region/VPC = recreate the group
* Lives OUTSIDE the EC2 → if traffic blocked, instance never even sees it
* Tip → keep 1 separate security group just for SSH access

#### Troubleshooting (exam trap)
* Timeout (connection hangs) → SECURITY GROUP issue
* "Connection refused" → security group worked, problem is the app (not launched/errored)

#### Referencing Security Groups (advanced)
* A security group can allow another security group in its inbound rules
* Instances with the right SG attached → talk to each other regardless of IP
* No IP juggling needed → common pattern with load balancers
* SG not listed in inbound rules → that instance is denied

#### Important Ports (memorize)
* 22 → SSH → log into Linux instance
* 22 → SFTP → secure file upload (over SSH)
* 21 → FTP → file transfer / upload to file share
* 80 → HTTP → unsecured websites
* 443 → HTTPS → secured websites (standard today)
* 3389 → RDP → log into Windows instance
* Remember → 22 = SSH (Linux), 3389 = RDP (Windows)