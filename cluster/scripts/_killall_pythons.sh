#!/bin/bash
#!/bin/bash
echo -e '\e[7m\e[93mmdl1\e[27m\e[39m'
ssh dl1 _killall_local_pythons.sh

echo -e '\e[7m\e[93mmdl2\e[27m\e[39m'
ssh dl2 _killall_local_pythons.sh

echo -e '\e[7m\e[93mmdl3\e[27m\e[39m'
ssh dl3 _killall_local_pythons.sh

echo -e '\e[7m\e[93mmdl4\e[27m\e[39m'
ssh dl4 _killall_local_pythons.sh
