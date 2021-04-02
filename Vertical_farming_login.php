<?php
$host="localhost";
$user="id16507032_user";
$pass="Vaibhavi@123";
$db="id16507032_vertical_farming";
$conn=mysqli_connect($host,$user,$pass,$db);
if(array_key_exists('disease_analysis', $_GET)) {
            button1();
        }
function button1() {
            header("Location:paari_dumma.php");
        }
if(isset($_GET['submit']))
{
	$name=$_GET['Username'];
	$phonenumber=$_GET['Email'];
	$password=$_GET['Password'];
	
	$sql="insert into Vertical_Registration(Username,Email,Password) values('$name','$phonenumber','$password')";
	
if(mysqli_query($conn,$sql))
{
	echo "success";
	


}
else
{
	echo "user already exists or not a valid phone number";
}
}
	?>
	
	
	





