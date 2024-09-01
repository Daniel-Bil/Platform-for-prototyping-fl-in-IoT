
function addLayer(Elements, Connections, Id){
    console.log("addLayer", Id)
    let bottomIds = returnBottomLayersId(Elements, Connections, Id)
    console.log("bottomIds", bottomIds)
    if(bottomIds.length==1){
        
        //Single normal connection return Layer + next Layer recurent
        let topIds = numberOfConnectionsTop(Elements, Connections, bottomIds[0])
        if (topIds>1){
            console.log("bottomIds.length==1>1")
            return [returnElementById(Elements, Id), bottomIds[0]]
        }
        else{
            console.log("bottomIds.length==1")
            return [returnElementById(Elements, Id),...addLayer(Elements, Connections, bottomIds[0])]
        }
    }

    if(bottomIds.length==0){
        console.log("%c bottomIds.length==0", 'background: #222; color: #bada55')
        //add only this layer cause no more connections or error xD
        return [returnElementById(Elements, Id)]
    }

    if(bottomIds.length>1){
        console.log("%c bottomIds.length>1", 'background: #222; color: #bada55')
        //add only this layer cause no more connections or error xD
        let parallelConnections = []
        let merger 
        for (let i=0;i<bottomIds.length;i++){
            console.log("now id = ", bottomIds[i])
            let bottomConnections = numberOfConnectionsBottom(Elements, Connections, bottomIds[i])
            if(bottomConnections>1){
                parallelConnections.push([])
            }
            else{
                let parallelConnection = [...addLayer(Elements, Connections, bottomIds[i])]
            
                console.log("parallelConnection1 = ", parallelConnection, merger)
                if (parallelConnection.length>1) {
                    merger = parallelConnection.pop()
                }
                else{
                    parallelConnection.pop()
                }

                console.log("parallelConnection2 = ", parallelConnection, merger, bottomConnections)
                parallelConnections.push(parallelConnection)
            }
            
            
            
        }
        console.log("parallelConnections = ", parallelConnections)
        return [returnElementById(Elements, Id), parallelConnections, ...addLayer(Elements, Connections, merger)]
    }
}

function returnElementById(Elements, Id){
    const index = Elements.findIndex(data => data.id === Id);
    if(index==-1){
        return 0
    }
    return Elements[index]
}

function FindLine(){

}


function numberOfConnectionsTop(Elements, Connections, Id){
    // checks number when given id is starting point
    

    let numberOfConnections = 0

    for(let i=0; i<Connections.length; i++){
        if(Connections[i].end===Id){
            numberOfConnections+=1
        }
    }
    console.log("numberOfConnectionsTop = ", numberOfConnections)
    return numberOfConnections
}

function numberOfConnectionsBottom(Elements, Connections, Id){
    // checks number when given id is ending point
    

    let numberOfConnections = 0

    for(let i=0; i<Connections.length; i++){
        if(Connections[i].end===Id){
            numberOfConnections+=1
        }
    }
    console.log("numberOfConnectionsBottom = ", numberOfConnections)
    return numberOfConnections
}

function returnBottomLayersId(Elements, Connections, Id){
    console.log("numberOfConnections")
    let layerIds = []

    for(let i=0; i<Connections.length; i++){
        if(Connections[i].start===Id){

            layerIds.push(Connections[i].end)
        }
    }
    return layerIds
}


function findIdOfFirstElement(Elements, Connections){
    // Function that with givven lists of Element and Connections between them is able to locate ID of first element
    
    console.log("findFirstElement")

    let numberOfFirst = 0
    let idOfFirst
    let tempId
    for(let i=0; i<Elements.length;i++){
        tempId = Elements[i].id
        const index = Connections.findIndex(data => data.end === tempId);
        console.log()
        if(index ==-1){
            numberOfFirst +=1
            idOfFirst = tempId
        }
    }
    console.log("number of first elements = ", numberOfFirst)
    console.log("id of first element = ", idOfFirst)
    if(numberOfFirst!=1){
        return -1
    }
    return idOfFirst


}

export function generateArchitectureJSON(Elements, Connections){
    console.log("TEST generateArchitectureJSON")
    let Id = findIdOfFirstElement(Elements, Connections)
    if (Id==-1){
        return -1
    }
    let architecture = []
    
    architecture.push(addLayer(Elements, Connections, Id))
    console.log("architecture = ", architecture)
    
    return architecture
}