### AWS Global Infrastructure
- AWS Regions
	- Named like us-east-1 eu-west-3
	- A cluster of data centers
	- Most aws services are region scoped
	- What factors effect choosing an aws region
		- Compliance: some time government want never go outside of their country
		- Latency
		- Not all regions have all services
			- ![[Pasted image 20260627161709.png]]
			  Here route 53 service => Says global => which mean this service present in all the regions 
		- Pricing changes per regions
- AWS Availability Zones
	- Named like if the region name is ap-southeast-2 -> then its availability zones name are -> ap-southeast-2a, ap-southeast-2b, ap-southeast-2c
	- Each regions have many availability zones (mostly 3-6) which are connected with high bandwidth ultra-low latency networking
	- Each availability zone has one or more data centers.
- AWS Data Centers
- AWS Edge Locations / Point
	- Used to deliver content to user with lower latency

### Global Service vs Non Global Service

| Global Service                                                                                                                                | vs Non Global Service                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| It operates from a **single global namespace** — you don't choose a region when creating it.                                                  | It operates from the **region** , which we need to pick                                           |
| ![[Pasted image 20260627162433.png]] <br>In the console, the region selector shows **"Global"** — meaning you can't even pick a region for it | ![[Pasted image 20260627162620.png]] In the console, the region selector shows **"regions name"** |
| For example: route 53                                                                                                                         | For example: Ec2                                                                                  |
