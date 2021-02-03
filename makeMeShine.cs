using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class makeMeShine : MonoBehaviour
{
	public GameObject mind;
	getThought mindCmp;

	private float nextActionTime = 0.0f;
 	private float period = 0.1f;

    // Start is called before the first frame update
    void Start()
    {
        mindCmp = mind.GetComponent<getThought>();
        GetComponent<ParticleSystem>().maxParticles = 0;
    }

    // Update is called once per frame
    void Update()
    {
    	if (Time.time > nextActionTime ) {
        	nextActionTime += period;
        }
        GetComponent<Light>().range = mindCmp.getMindPower() * 1.5f;
        GetComponent<Light>().intensity = mindCmp.getMindPower() * 1.5f;
        GetComponent<ParticleSystem>().maxParticles = (int) mindCmp.getMindPower() * 5;

    }
}
