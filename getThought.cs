using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class getThought : MonoBehaviour
{
	float mindPower;
	private float nextActionTime = 0.0f;
 	private float period = 0.2f;

 	private TcpClient socketConnection; 	
	private Thread clientReceiveThread;
 
    void Start()
    {
        mindPower = 0.0f;
        ConnectToTcpServer();
    }

    void Update()
{
		if (Time.time > nextActionTime) {
        	SendMessage();
        }
    }

    public float getMindPower()
    {
    	return mindPower;
    }

    void setMindPower(float newPower)
    {
    	mindPower = newPower;
    }

   private void ConnectToTcpServer () { 		
		try {  			
			clientReceiveThread = new Thread (new ThreadStart(ListenForData)); 			
			clientReceiveThread.IsBackground = true; 			
			clientReceiveThread.Start();  		
		} 		
		catch (Exception e) { 			
			Debug.Log("On client connect exception " + e); 		
		} 	
	}

	private void ListenForData() { 		
		try { 			
			socketConnection = new TcpClient("localhost", 60000);  			
			Byte[] bytes = new Byte[1024];             
			while (true) { 				
				// Get a stream object for reading 				
				using (NetworkStream stream = socketConnection.GetStream()) { 					
					int length; 					
					// Read incomming stream into byte arrary. 					
					while ((length = stream.Read(bytes, 0, bytes.Length)) != 0) { 						
						var incommingData = new byte[length]; 						
						Array.Copy(bytes, 0, incommingData, 0, length); 						
						// Convert byte array to string message. 						
						string serverMessage = Encoding.ASCII.GetString(incommingData); 						
						Debug.Log("server message received as: " + serverMessage);
						if (serverMessage == "go")
							mindPower += 0.1f;
						else if (mindPower > 0f)
							mindPower -= 0.1f;
					} 				
				} 			
			}         
		}         
		catch (SocketException socketException) {             
			Debug.Log("Socket exception: " + socketException);         
		}     
	}  	

	private void SendMessage() {         
		if (socketConnection == null) {             
			return;         
		}
		try { 			
			// Get a stream object for writing. 			
			NetworkStream stream = socketConnection.GetStream(); 			
			if (stream.CanWrite) {                 
				string clientMessage = "This is a message from one of your clients."; 				
				// Convert string message to byte array.                 
				byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes(clientMessage); 				
				// Write byte array to socketConnection stream.                 
				stream.Write(clientMessageAsByteArray, 0, clientMessageAsByteArray.Length);                 
				Debug.Log("Client sent his message - should be received by server");             
			}         
		} 		
		catch (SocketException socketException) {             
			Debug.Log("Socket exception: " + socketException);         
		}     
	}
}
