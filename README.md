What is Docker ??? 


First you need to understand Containers.

Containers are something like a virtual machine, it has its operating system and all, except that unlike virtual machines they don’t simulate the entire computer, but rather create a sand boxed environment that pretends to be a virtual machine.

Usually those containers also run the smallest possible version of the software, you can theoretically install an operating system like Ubuntu 20 with all its bells and whistles in a container, but what is typically done by professionals is that they build a stripped down version of the operating system that can do one job only.

Docker is a way to build and run those containers, save them into templates etc.

So for me, I have two containers running, one is running A redis server, and the other is running ArangoDB. Each of these is based on a build done by the respective companies, I have no idea what goes into building them, but I just start them with some minimal added configuration and they go. Each of these too are exposing only the ports needed to do their job on the network and nothing else. So a hacker can’t typically talk to them like a regular computer.

Now I am building an app that communicates with these two containers, putting data in and ou on their network interfaces.

When time comes to deploy my app to production, I don’t need to set up separate servers and what not. I just save my containers into image files then just go to Amazon and deploy with docker to the server instances I choose. I don’t care what operating system Amazon is running on those servers, because docker will build me containers that are identical to those I saved on my own computer and run them.

So it could be that Amazon I running Amazon linux but my container is running fedora, or Ubuntu, doesn’t matter, docker will handle the translation between the container and the host.

If get pissed off at Amazon and decide to go to Google cloud, I deploy the same files there, I don’t care what Google is using to run their servers, docker will do the translation.

Unlike virtual machines, because docker create these light weight containers, I won’t loose much performance.

Finally, a docker file is a simple instruction file that tells docker to download an image, then run some commands on it, such as install additional software, etc
