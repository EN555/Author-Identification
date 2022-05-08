import {useState,useEffect} from 'react';
import { Button,Form,FormGroup,FormControl } from 'react-bootstrap';
import { getInferencesInputExamples, infer } from '../services/api';
import CircularProgress from '@mui/material/CircularProgress';
import { truncate } from '../services/utils';
// import SendIcon from '@mui/icons-material/Send';


function InferForm() {
    const [inferencesExamples,setInferencesExamples] = useState([]);
    const [infering,setInfering] = useState(false);
    const [result,setResult] = useState("");
    const [text,setText] = useState();
    const [loadingExamples,setLoadingExamples] = useState(false);
    const [currExample,setCurrExample] = useState("");

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
        <div style={{marginTop: "1.5rem"}}> 
            <h1>Author Idenedication Demo</h1>
            <Form>
                <FormGroup className="mb-3" controlId="formBasicEmail">
                    <Form.Label>Select Input Example</Form.Label>
                    <Form.Control
                        as="select"
                        value={currExample}
                        onChange={e => {
                            setText(e.target.value);
                            setCurrExample(e.target.value);
                        }}
                    >
                        {inferencesExamples.map((example, index)=>(
                            <option key={index} value={example}>{truncate(example,20)}</option>
                        ))}
                    </Form.Control>
                    <Form.Label style={{marginTop: "1rem"}}>Text of Author</Form.Label>
                    <FormControl as="textarea" rows="5" type="text"
                        placeholder="Enter text"  
                        value={text} onChange={(e) => setText(e.target.value)}/>
                </FormGroup>
                <div style={{justifyContent:"center",display:"flex"}}>
                    {<p style={{display:"inline-block",flexGrow:100,fontWeight:"bold"}}>{result && `Predicted Label: ${result}`}</p>}
                    <div style={{paddingRight: "1rem",display: "inline-block"}}>
                        <Button variant="primary" onClick={!infering ? on_submit : null}>
                        {infering ? 'Loading...' : 'Send'}
                        </Button>
                    {infering && <CircularProgress size={30}/>}
                    </div>

                </div>
            </Form>
        </div>
    )
}

export default InferForm;