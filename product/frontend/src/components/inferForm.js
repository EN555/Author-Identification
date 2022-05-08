import {useState,useEffect} from 'react';
import { Button,Form,FormGroup,FormControl } from 'react-bootstrap';
import { getInferencesInputExamples, infer } from '../services/api';
import CircularProgress from '@mui/material/CircularProgress';
// import SendIcon from '@mui/icons-material/Send';


function InferForm() {
    const [inferencesExamples,setInferencesExamples] = useState([]);
    const [infering,setInfering] = useState(false);
    const [result,setResult] = useState("");
    const [text,setText] = useState();
    const [loadingExamples,setLoadingExamples] = useState(false);
    const on_submit = async()=>{
        console.log("infering...");
        setInfering(true);
        try{
            const res = await infer(JSON.stringify({"text":text}));
            setResult(res.data.author_name);
        }catch{}
        setInfering(false);
    }
    const featch_examples = async()=>{
        setLoadingExamples(true);
        try{
            const examples = await getInferencesInputExamples();
            setInferencesExamples(examples["data"]);
        }catch{}
        setLoadingExamples(false);
    }
    useEffect(()=>{
        featch_examples();
    },[]);
    return (
        <div> 
            <h1>Author Idenedication Demo</h1>
            <Form>
                <FormGroup className="mb-3" controlId="formBasicEmail">
                    <Form.Label>Text of Author</Form.Label>
                    <FormControl as="textarea" rows="3" type="text"
                        placeholder="Enter text"  
                        value={text} onChange={(e) => setText(e.target.value)}/>
                </FormGroup>
                <div style={{paddingLeft:"1rem",display:"inline-block",justifyContent:"center"}}>
                    {infering && <CircularProgress size={30}/>}
                    <div style={{paddingLeft: "1rem",display: "inline-block"}}>
                        <Button variant="primary" onClick={!infering ? on_submit : null}>
                        {infering ? 'Loading...' : 'Send'}
                        </Button>
                    </div>

                </div>
            </Form>
            {result && <div style={{marginTop: "1rem"}}>
                <h2>Result</h2>
                <p>predicted label is: {result}</p>
            </div>}
            {inferencesExamples.map((curr,index)=>(
                <p key={index}>curr</p>
            ))}
        </div>
    )
}

export default InferForm;