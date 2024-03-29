
"""
    The file data_task.json contains the mock data we were supposed to create according to the data already provided.
"""


def natural_language_to_sparql(natural_language_query_string):
    """
    Convert a natural language query to a SPARQL query.
    :param natural_language_query: The natural language query.
    :return: The SPARQL query.
    """

    """ PROBLEM:
        The problem is, that in the code, a package which is used tries to reference sklearn.ensemble.forest. 
        But that is deprecated. So the code needs to be updated to use the new package. I can not update the code of the package tho.
        And besides that, the API calls are deprecated as well, because the APIs are not responding.
        So I won't be able to test out, which will work and which won't. But I still want to explain the general
        idea what we would go for to solve the task.
    """

    #IDEA

    # After Training the Model, the test Files are the ones, which allow us to test it. 
    # There they will first load in the data which have been generated by training, and then they will read the data, that only has the questions in it.
    # For those questions, multiple answers will be generated. The generated queries (which are the answers) will be sorted.
    # The best answer will be the one, which has the highest score.
    # After picking the best answer, the model is rated in the test files according to precision, recall and f1 score.

    # What would we do to generate a query, for given natural language?
    # According to the task we have to do it according to the data created in Task 2 c)
    # We would use the data to train a model, which is able to generate queries for given natural language.

    # So we need to preprocess the dataset, and generate linked_answers.json.
    # After that we will generate the output file with the queries, which are generated according to extracted entitites and relations.
    # all of that is to train the model with the newly generated data. 
    # after splitting the data into training, trial and test data, the dependency tree will be generated.
    # Part of that is the needed input and output files, which are used to train the model.
    # Then the phrase mappings are generated, which are mapping of entities and relationships. 

    # Afterwords the queries can be generated, and tested.
    # In this case, the testing will of the queries afterwords and calculating recall, precision and f1 score is unnecessary. 
    # Because the goal is only to generate the Sparql Query not test if the results match up with pre calculated ones, which we don't have in this case.

    # Do calculate a Sparql Query, after traingin and preparing the model, we need a linker, a parser, a question_type_classifier, and a knowledge base.
    # The Knowledge Base and Parser, can be extracted by the linked_test.json.
    # The question type classifier is a result, that has been put into output directory before. 
    # The linker is a result of the generated phrase mappings. 
    # In the given code, not only given Knowledge base is used, but also a online knowledge base. The problem is that these weren't able to be tested because the API calls are not working in this project.
    
    # from the linker we extract entities and ontologies. If there is a (or multiple) answer(s) in the question answer pair, we will iterate through the entities. 
    # In those entities, we will generate the subsets, and same think for the ontologies. Then the Entities and Ontologies will be combined into one list of tuples.
    # Now the questions will be passed into the generate_query method, with the entity and ontology. 
    # THere the question type is classified. 
    # Afterwards the graph is being build of the knowledge base, and a minimal subgraph is being found with given entities and relations. Then out of that graph, a query builder is run over. 
    # It will generate possible queries (or in the code referred to as valid walks, which are the paths in the graph).
    # the rank method is then used to rank those walks. 
    # In the case of a good rank, the queries will be returned. 
    # The where questions are filtered out, and the one with the highest calculated score/confidence is returned. 



    return natural_language_query_string