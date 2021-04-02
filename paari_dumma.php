<html>
<head>
</head>
<body>
<?php
$res=shell_exec("python disease_ten.py");
$states=explode("\n",$res);


echo "$states[6]";


?>
</body>
</html>